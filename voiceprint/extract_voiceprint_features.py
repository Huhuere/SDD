"""提取声纹（Sinc-MSA）预训练后的固定维度表示。

使用说明：
1. 先运行 pretrain_contrastive_voiceprint.py 得到每个 fold 的最佳权重 (pretrain_voiceprint_foldX_best.pth)。
2. 决定要用哪个 fold 的权重（或平均/集成自行处理），指定 --checkpoint。
3. 提供一个 .scp 列表 (每行一个音频文件，相对 data_folder 或绝对路径)。
4. 本脚本对每个文件：滑窗划分 (窗口长度 wlen = fs*cw_len/1000, 步长 wshift = fs*cw_shift/1000)，得到多个 chunk，取模型中间表示 (fc_audio1 输出/tsne_data) 的平均作为整段音频的声纹 embedding。
5. 输出：
    - voice_embeddings.npy  (N, D)
    - filenames.txt         (N 行，对应音频顺序)
    - per_file/*.npy        每个音频的单独向量

默认 D = hidden_dims_fc (来自 cfg 里的 hidden_dims_fc)。

运行示例：
    python voiceprint/extract_voiceprint_features.py \
        --cfg voiceprint/cfg/5fold_train_up.cfg \
        --scp lists/test0.scp \
        --data-folder <your_wav_root_dir> \
        --checkpoint pretrain_ckpts_voiceprint/pretrain_voiceprint_fold0_best.pth \
        --output-dir voiceprint_features_fold0
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from tqdm import tqdm

from dnn_models import SincNet_attention_gnn as VoiceModel
from data_io_attention import read_conf, ReadList, str_to_bool


def build_model_from_cfg(cfg):
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
        'input_dim': int(cfg.fs) * int(cfg.cw_len) // 1000,
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
        'num_classes': int(cfg.num_classes),
    }
    model = VoiceModel(arch)
    return model, arch


def sliding_windows(signal: np.ndarray, wlen: int, wshift: int):
    if len(signal) < wlen:
        pad = wlen - len(signal)
        signal = np.concatenate([signal, np.zeros(pad, dtype=np.float32)])
    windows = []
    beg = 0
    while beg + wlen <= len(signal):
        windows.append(signal[beg:beg + wlen])
        beg += wshift
    if not windows:  # 极短
        windows.append(signal[:wlen])
    return np.stack(windows)


def extract_for_file(path: str, model, wlen: int, wshift: int, batch_win: int = 128):
    wav, fs = librosa.load(path, sr=None, mono=True)
    wins = sliding_windows(wav, wlen, wshift)  # (Nw, wlen)
    reps = []
    with torch.no_grad():
        for i in range(0, wins.shape[0], batch_win):
            chunk = torch.from_numpy(wins[i:i + batch_win]).float().cuda()
            _, rep = model(chunk)
            reps.append(rep.detach().cpu().numpy())
    reps = np.concatenate(reps, axis=0)
    emb = reps.mean(axis=0)  # 平均池化
    return emb


def parse_args():
    ap = argparse.ArgumentParser(description='提取预训练 Sinc-MSA 声纹表征')
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--scp', required=True, help='.scp 列表文件')
    ap.add_argument('--data-folder', required=True, help='音频根目录(与训练时 data_folder 保持一致) 末尾不要漏 / 或 \\')
    ap.add_argument('--checkpoint', required=True, help='pretrain_voiceprint_foldX_best.pth 文件路径')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--batch-win', type=int, default=128, help='窗口批处理大小')
    ap.add_argument('--cpu', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    # 这里与预训练脚本一致：临时伪造 argv 只保留 --cfg，防止报 "no such option"。
    orig_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], '--cfg', args.cfg]
        cfg = read_conf(args.cfg)
    finally:
        sys.argv = orig_argv

    # 清理 data_folder（可能含有内联注释或尾部空格）。
    data_folder = args.data_folder.strip()
    if '#' in data_folder:
        data_folder = data_folder.split('#', 1)[0].strip()
    if not os.path.isdir(data_folder):
        print(f"[WARN] data_folder 路径不存在: {data_folder}")
    else:
        print(f"[INFO] 使用音频根目录: {data_folder}")

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model, arch = build_model_from_cfg(cfg)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state = ckpt.get('model_state', ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    wlen = arch['input_dim']
    wshift = int(arch['fs'] * int(cfg.cw_shift) / 1000) if hasattr(cfg, 'cw_shift') else wlen  # 若未在 cfg 中提供 cw_shift 则不滑动
    file_list = [l.strip() for l in ReadList(args.scp) if l.strip()]
    os.makedirs(args.output_dir, exist_ok=True)
    per_dir = os.path.join(args.output_dir, 'per_file')
    os.makedirs(per_dir, exist_ok=True)
    embeddings = []
    names = []
    for fn in tqdm(file_list, desc='Extract'):
        # 支持 .scp 行里已经是绝对路径；否则与 data_folder 拼接
        if os.path.isabs(fn):
            full_path = fn
        else:
            full_path = os.path.join(data_folder, fn)
        if not os.path.isfile(full_path):
            print(f"[WARN] 文件不存在，跳过: {full_path}")
            continue
        emb = extract_for_file(full_path, model, wlen, wshift, batch_win=args.batch_win)
        embeddings.append(emb)
        names.append(fn)
        np.save(os.path.join(per_dir, os.path.splitext(os.path.basename(fn))[0] + '.npy'), emb.astype(np.float32))
    if len(embeddings) == 0:
        print('[ERROR] 未成功处理任何文件，请检查 --scp 与 --data-folder 是否匹配。')
        return
    embeddings = np.stack(embeddings).astype(np.float32)
    np.save(os.path.join(args.output_dir, 'voice_embeddings.npy'), embeddings)
    np.savetxt(os.path.join(args.output_dir, 'filenames.txt'), names, fmt='%s')
    print(f'[DONE] 提取完成: {embeddings.shape[0]} 个文件, 维度={embeddings.shape[1]} 保存到 {args.output_dir}')


if __name__ == '__main__':
    main()

