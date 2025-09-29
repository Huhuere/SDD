"""Contrastive pretraining for Sinc-MSA (SincNet + Multi-Head Self-Attention) voiceprint backbone.

逻辑说明（中文）：
1. 目的：先在每个折(fold)的训练数据上，用对比学习(contrastive learning, NT-Xent / InfoNCE)预训练声纹表示，使 Sinc + MSA 学到纯声纹判别特征，不受下游多模态标签限制。
2. 训练策略：
   - 每个 batch 从训练列表随机采样若干音频，针对每条音频随机裁剪两段（或两种数据增强后的片段）作为正样本对 (view1, view2)。
   - 所有不同音频之间的样本视为负样本。
   - 使用温度缩放的 NT-Xent 损失函数：最大化同一音频两视图余弦相似度，最小化不同音频间相似度。
3. 模型：复用 SincNet_attention_gnn，只取其中返回的表征 (tsne_data)。不使用分类头的输出 logits。
4. 投影头（Projection Head）：常规 SimCLR 做法会再加一层/两层 MLP 做到 128 维；这里默认添加 Linear -> ReLU -> Linear 到 128 维，可通过参数关闭。
5. 优化器：RMSprop，lr=0.03（可调），epochs=200（可调），batch_size=256（可调）。
6. 每折都会单独训练并保存最佳(最小 loss)权重：pretrain_voiceprint_fold{fold_i}_best.pth
7. 后续融合阶段：加载该 checkpoint，只冻结 conv(Sinc) + mul_attention 模块的参数即可：
       for p in model.conv.parameters(): p.requires_grad = False
       for p in model.mul_attention.parameters(): p.requires_grad = False
   （可选）若希望完全固定声纹表示，也可再冻结 fc_audio1。

运行最简示例：
    python voiceprint/pretrain_contrastive_voiceprint.py \
        --cfg voiceprint/cfg/5fold_train_up.cfg \
        --tr-list-prefix lists/train \
        --folds 5 \
        --save-dir pretrain_ckpts_voiceprint \
        --epochs 200 --batch-size 256 --lr 0.03

作者：自动生成脚本 (2025)
"""

import os
import math
import argparse
import random
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import librosa

from dnn_models import SincNet_attention_gnn as VoiceModel
from data_io_attention import read_conf, ReadList, str_to_bool


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model_from_cfg(cfg):
    """根据已有 cfg 字段构建和 train_5fold 一致的 Sinc + MSA 模型"""
    cnn_N_filt = list(map(int, cfg.cnn_N_filt.split(',')))
    cnn_len_filt = list(map(int, cfg.cnn_len_filt.split(',')))
    cnn_max_pool_len = list(map(int, cfg.cnn_max_pool_len.split(',')))
    cnn_use_laynorm_inp = str_to_bool(cfg.cnn_use_laynorm_inp)
    cnn_use_batchnorm_inp = str_to_bool(cfg.cnn_use_batchnorm_inp)
    cnn_use_laynorm = list(map(str_to_bool, cfg.cnn_use_laynorm.split(',')))
    cnn_use_batchnorm = list(map(str_to_bool, cfg.cnn_use_batchnorm.split(',')))
    cnn_act = list(map(str, cfg.cnn_act.split(',')))
    cnn_drop = list(map(float, cfg.cnn_drop.split(',')))

    arch = {
        'input_dim': int(cfg.fs) * int(cfg.cw_len) // 1000,  # wlen (samples)
        'fs': int(cfg.fs),
        'cnn_N_filt': cnn_N_filt,
        'cnn_len_filt': cnn_len_filt,
        'cnn_max_pool_len': cnn_max_pool_len,
        'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
        'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
        'cnn_use_laynorm': cnn_use_laynorm,
        'cnn_use_batchnorm': cnn_use_batchnorm,
        'cnn_act': cnn_act,
        'cnn_drop': cnn_drop,
        'mulhead_num_hiddens': int(cfg.mulhead_num_hiddens),
        'mulhead_num_heads': int(cfg.mulhead_num_heads),
        'mulhead_num_query': int(cfg.mulhead_num_query),
        'dropout_fc': float(cfg.dropout_fc),
        'hidden_dims_fc': int(cfg.hidden_dims_fc),
        'att_hidden_dims_fc': int(cfg.att_hidden_dims_fc),
        'num_classes': int(cfg.num_classes),  # 预训练不使用分类 head，但结构保持一致
    }
    model = VoiceModel(arch)
    return model, arch


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128, hidden: int = 0):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, proj_dim)
            )
        else:
            self.net = nn.Linear(in_dim, proj_dim)

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)


def contrastive_batch(wav_list, data_folder, wlen, batch_size, amp_jitter=0.2):
    """生成两组视图 (view1, view2) 以及文件索引。随机裁剪 + 幅度扰动。
    返回: v1, v2  (batch, wlen)
    """
    N = len(wav_list)
    indices = np.random.randint(N, size=batch_size)
    v1 = np.zeros((batch_size, wlen), dtype=np.float32)
    v2 = np.zeros((batch_size, wlen), dtype=np.float32)
    for i, idx in enumerate(indices):
        path = os.path.join(data_folder, wav_list[idx])
        wav, fs = librosa.load(path, sr=None, mono=True)
        if len(wav) < wlen + 10:
            pad = wlen + 10 - len(wav)
            wav = np.concatenate([wav, np.zeros(pad, dtype=np.float32)])
        # 两次独立随机裁剪
        def random_crop(arr):
            max_beg = len(arr) - wlen
            beg = 0 if max_beg <= 0 else np.random.randint(0, max_beg)
            return arr[beg:beg + wlen]
        seg1 = random_crop(wav)
        seg2 = random_crop(wav)
        # 幅度扰动
        a1 = np.random.uniform(1 - amp_jitter, 1 + amp_jitter)
        a2 = np.random.uniform(1 - amp_jitter, 1 + amp_jitter)
        v1[i] = seg1 * a1
        v2[i] = seg2 * a2
    v1 = torch.from_numpy(v1).float().cuda()
    v2 = torch.from_numpy(v2).float().cuda()
    return v1, v2


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """标准 NT-Xent (SimCLR) 损失。
    z1, z2: (B, D)  已经 L2 normalize
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)  # (2B,2B)
    # 构造 mask，去掉自身
    self_mask = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
    sim = sim / temperature

    # positives: (i, i+B) and (i+B, i)
    positives = torch.cat([
        F.cosine_similarity(z1, z2, dim=-1),
        F.cosine_similarity(z2, z1, dim=-1)
    ], dim=0) / temperature  # (2B,)

    # 对每一行做 softmax： log_sum_exp over negatives+positives (排除 self)
    sim_exp = torch.exp(sim.masked_fill(self_mask, float('-inf')))
    denom = sim_exp.sum(dim=1)

    loss = -torch.log(torch.exp(positives) / denom)
    return loss.mean()


def save_checkpoint(save_dir, fold_i, epoch, model, proj_head, best=False):
    os.makedirs(save_dir, exist_ok=True)
    fn = f'pretrain_voiceprint_fold{fold_i}_epoch{epoch}.pth'
    if best:
        fn = f'pretrain_voiceprint_fold{fold_i}_best.pth'
    path = os.path.join(save_dir, fn)
    torch.save({
        'model_state': model.state_dict(),
        'proj_head_state': None if proj_head is None else proj_head.state_dict(),
        'epoch': epoch,
    }, path)
    return path


def run_fold(fold_i, args, cfg):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model, arch = build_model_from_cfg(cfg)
    model.to(device)

    if args.proj_dim > 0:
        proj_head = ProjectionHead(in_dim=arch['hidden_dims_fc'], proj_dim=args.proj_dim, hidden=args.proj_hidden).to(device)
    else:
        proj_head = None

    optimizer = torch.optim.RMSprop(
        list(model.parameters()) + ([] if proj_head is None else list(proj_head.parameters())),
        lr=args.lr, alpha=0.95, eps=1e-8
    )

    # 读取当前 fold 的训练列表
    train_scp = f"{args.tr_list_prefix}{fold_i}.scp"
    wav_list = ReadList(train_scp)
    # 去除每行可能的多余空白
    wav_list = [w.strip() for w in wav_list if w.strip()]
    print(f"[Fold {fold_i}] Train files: {len(wav_list)} from {train_scp}")

    # 清理 cfg 中 data_folder 里可能残留的注释，防止出现 "path  # comment" 被当成真实目录
    raw_data_folder = getattr(cfg, 'data_folder', getattr(cfg, 'data_dir', ''))
    data_folder = raw_data_folder.strip()
    if '#' in data_folder:
        # 只取 # 之前的有效路径部分
        data_folder = data_folder.split('#', 1)[0].strip()
    if not data_folder:
        raise ValueError(f"配置文件中 data_folder 为空（原值: '{raw_data_folder}'），请在 cfg 中设置 data_folder=音频根目录")
    if not os.path.isdir(data_folder):
        print(f"[WARN] data_folder 路径不存在: '{data_folder}' (原始: '{raw_data_folder}')")
    else:
        print(f"[INFO] 使用音频根目录: {data_folder}")

    wlen = arch['input_dim']  # 样本点长度

    best_loss = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        if proj_head is not None:
            proj_head.train()
        epoch_loss = 0.0
        steps = args.steps_per_epoch if args.steps_per_epoch > 0 else max(1, len(wav_list) // args.batch_size)
        for step in range(steps):
            v1, v2 = contrastive_batch(wav_list, data_folder + '/', wlen, args.batch_size, amp_jitter=args.amp_jitter)
            # 前向；获取表示 (tsne_data)
            logits1, rep1 = model(v1)
            logits2, rep2 = model(v2)
            if proj_head is not None:
                z1 = proj_head(rep1)
                z2 = proj_head(rep2)
            else:
                # 若无投影头，直接归一化
                z1 = F.normalize(rep1, dim=-1)
                z2 = F.normalize(rep2, dim=-1)
            loss = nt_xent_loss(z1, z2, temperature=args.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / steps
        print(f"[Fold {fold_i}] Epoch {epoch}/{args.epochs} ContrastiveLoss={avg_loss:.4f}")
        # 保存普通 checkpoint（可选）
        if epoch % args.ckpt_interval == 0 or epoch == args.epochs:
            save_checkpoint(args.save_dir, fold_i, epoch, model, proj_head, best=False)
        # 记录最佳
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = save_checkpoint(args.save_dir, fold_i, epoch, model, proj_head, best=True)
            print(f"    * New best loss {best_loss:.4f} -> {best_path}")

    print(f"[Fold {fold_i}] Done. Best loss={best_loss:.4f}")


def parse_args():
    ap = argparse.ArgumentParser(description="Pretrain Sinc-MSA voiceprint encoder with contrastive learning (NT-Xent)")
    ap.add_argument('--cfg', required=True, help='路径：voiceprint/cfg/*.cfg (复用原有模型结构超参)')
    ap.add_argument('--tr-list-prefix', required=True, help='训练列表前缀，如 lists/train (脚本会拼接 fold_idx+.scp)')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--save-dir', default='pretrain_ckpts_voiceprint')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=0.03)
    ap.add_argument('--temperature', type=float, default=0.07)
    ap.add_argument('--proj-dim', type=int, default=128, help='投影维度，<=0 则不用投影头，直接使用原表征')
    ap.add_argument('--proj-hidden', type=int, default=0, help='投影 MLP 隐藏层，0=单线性层')
    ap.add_argument('--steps-per-epoch', type=int, default=0, help='0=按数据量估算；否则强制每 epoch 步数')
    ap.add_argument('--ckpt-interval', type=int, default=50, help='多少个 epoch 额外保存一次 checkpoint')
    ap.add_argument('--amp-jitter', type=float, default=0.2, help='幅度扰动范围 ±值')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true', help='仅用 CPU')
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    # read_conf 内部使用 OptionParser 再次解析 sys.argv，只接受 --cfg；
    # 这里临时伪造 argv，避免我们新增的其它参数被它报错。
    orig_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], '--cfg', args.cfg]
        cfg = read_conf(args.cfg)
    finally:
        sys.argv = orig_argv
    print('[INFO] Start contrastive pretraining with params:')
    for k, v in vars(args).items():
        print(f'  {k}: {v}')
    for fold_i in range(args.folds):
        run_fold(fold_i, args, cfg)
    print('[DONE] All folds contrastive pretraining finished.')


if __name__ == '__main__':
    main()
