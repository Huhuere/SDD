"""Build PyG graph dataset (.pt lists) for fusion model.

example:
    python fusion/build_graph_datasets.py \
        --voiceprint-dir voiceprint_features_fold0 \
        --emotion-dir features_emotion_daic_segments \ 
        --pause-dir pause --energy-dir energy --tremor-dir tremor \
        --train-prefix lists/train --val-prefix lists/val --test-prefix lists/test \
        --folds 5 \
        --num-att-features 256 --num-pause-input 60 --num-enegy-features 100 --num-tromer-features 100 \
        --output-dir .

输出: data_train_{k}.pt / data_val_{k}.pt / data_test_{k}.pt

"""

import os
import argparse
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm
import torch
from torch_geometric.data import Data


def read_list(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines


def load_labels(labels_file: str) -> Dict[str, int]:
    mapping = {}
    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ',' in line:
                parts = line.split(',')
            else:
                parts = line.split()
            if len(parts) < 2:
                continue
            base = parts[0]
            lab = int(parts[1])
            mapping[base] = lab
    return mapping


def build_basename(filename: str) -> str:
    fn = os.path.basename(filename)
    if fn.endswith('.wav'):
        fn = fn[:-4]
    return fn


def load_voiceprint(voice_dir: str) -> Dict[str, np.ndarray]:
    """加载声纹嵌入。

    首选结构:
        voice_dir/
            voice_embeddings.npy
            filenames.txt (与 npy 行对应)
            per_file/*.npy (可选)

    兼容结构 (用户传入 --voiceprint-dir=voiceprint_features_foldX/per_file):
        <dir_with_only_individual_embeddings>/*.npy

    若找不到 voice_embeddings.npy，则自动扫描:
        1) 若存在 voice_dir/per_file 目录 -> 使用其中的 *.npy
        2) 否则直接扫描 voice_dir 下的 *.npy
    """
    emb_path = os.path.join(voice_dir, 'voice_embeddings.npy')
    name_path = os.path.join(voice_dir, 'filenames.txt')
    out: Dict[str, np.ndarray] = {}
    if os.path.isfile(emb_path) and os.path.isfile(name_path):
        embs = np.load(emb_path)
        names = read_list(name_path)
        if len(names) != embs.shape[0]:
            raise ValueError("voice_embeddings 与 filenames.txt 行数不一致")
        for n, v in zip(names, embs):
            out[build_basename(n)] = v.astype(np.float32)
        return out

    # Fallback 扫描 per_file 或当前目录
    scan_dir = None
    per_dir = os.path.join(voice_dir, 'per_file')
    if os.path.isdir(per_dir):
        scan_dir = per_dir
    else:
        scan_dir = voice_dir
    scanned = 0
    for fn in os.listdir(scan_dir):
        if not fn.endswith('.npy'):
            continue
        if fn == 'voice_embeddings.npy':  # 已在前面处理过常规模式
            continue
        path = os.path.join(scan_dir, fn)
        try:
            arr = np.load(path)
        except Exception as e:
            print(f"[WARN] 加载声纹文件失败 {path}: {e}")
            continue
        if arr.ndim != 1:
            print(f"[WARN] 声纹向量不是 1D, 跳过 {path}, shape={arr.shape}")
            continue
        base = fn[:-4]
        out[base] = arr.astype(np.float32)
        scanned += 1
    if scanned == 0:
        raise FileNotFoundError(f"未找到 voice_embeddings.npy 且未在 {scan_dir} 发现任何单文件嵌入 .npy")
    print(f"[INFO] 未找到 voice_embeddings.npy, 已从 {scan_dir} 直接扫描 {scanned} 个单文件声纹向量")
    return out


def load_emotion(emotion_dir: str) -> Dict[str, np.ndarray]:
    # 支持两种结构: 1) 直接是 per-file 向量 *.npy  2) segments/*.npy 需要聚合
    per_file_dir = emotion_dir
    segments_dir = os.path.join(emotion_dir, 'segments')
    result = {}
    if os.path.isdir(segments_dir):
        # 聚合 segments
        group = defaultdict(list)
        for fn in os.listdir(segments_dir):
            if not fn.endswith('.npy'):
                continue
            arr = np.load(os.path.join(segments_dir, fn))
            # 去掉 _segXXX 部分
            base = fn
            if '_seg' in base:
                base = base.split('_seg')[0]
            if base.endswith('.npy'):
                base = base[:-4]
            group[base].append(arr)
        for base, vecs in group.items():
            result[base] = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
    else:
        # 直接读取文件
        for fn in os.listdir(per_file_dir):
            if fn.endswith('.npy'):
                base = fn[:-4]
                result[base] = np.load(os.path.join(per_file_dir, fn)).astype(np.float32)
    return result


def load_scalar_txt_dir(path_dir: str, expected_first_dim: int = None) -> Dict[str, np.ndarray]:
    out = {}
    if not os.path.isdir(path_dir):
        return out
    for fn in os.listdir(path_dir):
        if not (fn.endswith('.txt') or fn.endswith('.npy')):
            continue
        base = fn.rsplit('.', 1)[0]
        p = os.path.join(path_dir, fn)
        if fn.endswith('.npy'):
            arr = np.load(p)
        else:
            arr = np.loadtxt(p)
        arr = np.array(arr, dtype=np.float32)
        # energy / tremor 可能是 (2, T) 或 (T, 2)
        if arr.ndim == 2 and arr.shape[0] != 2 and arr.shape[1] == 2:
            arr = arr.T
        out[base] = arr
    return out


def repeat_pause_vector(pause_vec: np.ndarray, num_pause_input: int) -> np.ndarray:
    # pause_vec: shape (6,) or (6,1) -> flatten
    v = pause_vec.flatten()
    if v.shape[0] != 6:
        raise ValueError(f"pause 特征应为 6 维, got {v.shape}")
    repeat_times = num_pause_input // 6
    tiled = np.tile(v, repeat_times)
    return tiled.astype(np.float32)


def fit_or_pad_sequence(two_by_T: np.ndarray, target_T: int) -> np.ndarray:
    """将任意输入规范成 (2, target_T)。

    允许以下异常情况自动修复:
    - shape == (0,) : 用全零 (2,target_T)
    - 1D 向量 (T,) : 视为单通道，复制或补零成 2 通道
    - (T,2) : 自动转置
    - (1,T) : 复制为 (2,T)
    - (2,T') : 裁剪/填充到 target_T
    其它形状仍报错，避免 silently 错误。
    """
    arr = two_by_T
    # 空
    if arr.size == 0:
        return np.zeros((2, target_T), dtype=np.float32)
    # 1D
    if arr.ndim == 1:
        # (T,) -> (2,T) 第二通道置零
        arr = arr.astype(np.float32)
        arr = np.stack([arr, np.zeros_like(arr)], axis=0)
    elif arr.ndim == 2:
        if arr.shape[0] != 2 and arr.shape[1] == 2:
            arr = arr.T  # (T,2) -> (2,T)
        elif arr.shape[0] == 1:  # (1,T) 复制
            arr = np.repeat(arr, 2, axis=0)
    else:
        raise ValueError(f"无法处理序列形状: {arr.shape}")

    if arr.shape[0] != 2:
        raise ValueError(f"期望 shape=(2,T), got {arr.shape}")

    T = arr.shape[1]
    if T == target_T:
        return arr.astype(np.float32)
    if T > target_T:
        start = (T - target_T) // 2
        return arr[:, start:start+target_T].astype(np.float32)
    pad_need = target_T - T
    last = arr[:, -1:]
    pad_frames = np.repeat(last, pad_need, axis=1)
    return np.concatenate([arr, pad_frames], axis=1).astype(np.float32)


def build_sample_tensor(voice_vec: np.ndarray,
                        emo_vec: np.ndarray,
                        pause_vec_flat: np.ndarray,
                        energy_2xt: np.ndarray,
                        tremor_2xt: np.ndarray,
                        num_att_features: int,
                        num_pause_input: int,
                        num_enegy_features: int,
                        num_tromer_features: int) -> np.ndarray:
    """
    返回形状 (5, 2*num_att_features) -> Data.x (后续 DataLoader 拼接后变 (B*5, 2*num_att_features))
    """
    sample = np.zeros((5, 2 * num_att_features), dtype=np.float32)

    def place(row: int, ch: int, vec: np.ndarray):
        # vec 放入 sample[row, ch*num_att_features : ch*num_att_features + len(vec)]
        start = ch * num_att_features
        sample[row, start:start+len(vec)] = vec[:min(len(vec), num_att_features)]

    # 0 attsinc
    place(0, 0, voice_vec)
    # 1 emotion
    place(1, 0, emo_vec)
    # 2 pause (only channel 0, needs length num_pause_input)
    place(2, 0, pause_vec_flat)
    # 3 energy (two channels each length num_enegy_features)
    place(3, 0, energy_2xt[0, :num_enegy_features])
    place(3, 1, energy_2xt[1, :num_enegy_features])
    # 4 tremor
    place(4, 0, tremor_2xt[0, :num_tromer_features])
    place(4, 1, tremor_2xt[1, :num_tromer_features])
    return sample


def fully_connected_5_nodes(include_self_loops=False):
    src = []
    dst = []
    for i in range(5):
        for j in range(5):
            if i != j or include_self_loops:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index


def build_dataset(list_file: str,
                  voice_dict: Dict[str, np.ndarray],
                  emo_dict: Dict[str, np.ndarray],
                  pause_dict: Dict[str, np.ndarray],
                  energy_dict: Dict[str, np.ndarray],
                  tremor_dict: Dict[str, np.ndarray],
                  labels: Dict[str, int],
                  args) -> List[Data]:
    names = read_list(list_file)
    data_objs = []
    miss_stats = defaultdict(int)
    edge_index = fully_connected_5_nodes()
    for name in tqdm(names, desc=f'Build {os.path.basename(list_file)}'):
        base = build_basename(name)
        if base not in labels:
            miss_stats['label'] += 1
            continue
        if base not in voice_dict:
            miss_stats['voice'] += 1; continue
        if base not in emo_dict:
            miss_stats['emotion'] += 1; continue
        if base not in pause_dict:
            miss_stats['pause'] += 1; continue
        if base not in energy_dict:
            miss_stats['energy'] += 1; continue
        if base not in tremor_dict:
            miss_stats['tremor'] += 1; continue

        voice_vec = voice_dict[base]
        emo_vec = emo_dict[base]
        pause_vec = repeat_pause_vector(pause_dict[base], args.num_pause_input)
        energy_seq = fit_or_pad_sequence(energy_dict[base], args.num_enegy_features)
        tremor_seq = fit_or_pad_sequence(tremor_dict[base], args.num_tromer_features)

        sample_arr = build_sample_tensor(voice_vec, emo_vec, pause_vec, energy_seq, tremor_seq,
                                         args.num_att_features, args.num_pause_input,
                                         args.num_enegy_features, args.num_tromer_features)
        x = torch.from_numpy(sample_arr)  # (5, 2*num_att_features)
        y = torch.tensor(labels[base]).long()
        data = Data(x=x, edge_index=edge_index, y=y)
        # 保存名字 (供验证阶段 person 统计)
        data.name_my = [[base]]  # 保持原脚本的索引方式
        data_objs.append(data)
    if miss_stats:
        print(f'[WARN] 缺失统计 ({list_file}): ' + ', '.join([f'{k}={v}' for k,v in miss_stats.items()]))
    print(f'完成 {list_file}: usable samples={len(data_objs)}')
    return data_objs


def derive_labels_from_lists(all_list_files: List[str]) -> Dict[str, int]:
    label_map = {}
    for lf in all_list_files:
        if not lf or not os.path.isfile(lf):
            continue
        for line in read_list(lf):
            base = build_basename(line)
            if base in label_map:
                continue
            # 规则: 最后一个 '_' 分隔的 token 为数字标签
            parts = base.split('_')
            if not parts:
                continue
            last = parts[-1]
            if last.isdigit():
                label_map[base] = int(last)
            else:
                # 忽略无法解析的 (可改为 raise)
                pass
    return label_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--voiceprint-dir', required=True)
    ap.add_argument('--emotion-dir', required=True)
    ap.add_argument('--pause-dir', required=True)
    ap.add_argument('--energy-dir', required=True)
    ap.add_argument('--tremor-dir', required=True)
    ap.add_argument('--labels', required=False, help='basename,label 格式 (CSV 或空白分隔); 若不提供则从文件名末尾数字自动推断')
    ap.add_argument('--train-prefix', required=True, help='如 lists/train')
    ap.add_argument('--val-prefix', required=True, help='如 lists/val')
    ap.add_argument('--test-prefix', default=None, help='如 lists/test (可选)')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--output-dir', default='.')
    # dim settings (确保与 cfg 对齐)
    ap.add_argument('--num-att-features', type=int, default=256)
    ap.add_argument('--num-pause-input', type=int, default=60)
    ap.add_argument('--num-enegy-features', type=int, default=100)
    ap.add_argument('--num-tromer-features', type=int, default=100)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('[1] 加载 voiceprint...')
    voice_dict = load_voiceprint(args.voiceprint_dir)
    print('[2] 加载 emotion...')
    emo_dict = load_emotion(args.emotion_dir)
    print('[3] 加载 pause...')
    pause_dict = load_scalar_txt_dir(args.pause_dir)
    print('[4] 加载 energy...')
    energy_dict = load_scalar_txt_dir(args.energy_dir)
    print('[5] 加载 tremor...')
    tremor_dict = load_scalar_txt_dir(args.tremor_dir)
    print('[6] 处理标签 ...')
    labels = {}
    if args.labels:
        labels = load_labels(args.labels)
        print(f'    从 {args.labels} 读取标签: {len(labels)} 条')
    else:
        # 汇总所有列表文件用于自动推断
        all_lists = []
        for k in range(args.folds):
            all_lists.append(f'{args.train_prefix}{k}.scp')
            all_lists.append(f'{args.val_prefix}{k}.scp')
            if args.test_prefix:
                all_lists.append(f'{args.test_prefix}{k}.scp')
        labels = derive_labels_from_lists(all_lists)
        print(f'    未提供 --labels，已根据文件名末尾数字自动推断 {len(labels)} 条标签 (规则: basename 最后一个 "_" 后为数字)')

    for k in range(args.folds):
        train_list = f'{args.train_prefix}{k}.scp'
        val_list = f'{args.val_prefix}{k}.scp'
        if not os.path.isfile(train_list):
            print(f'[Fold {k}] 缺少 {train_list}, 跳过该 fold')
            continue
        if not os.path.isfile(val_list):
            print(f'[Fold {k}] 缺少 {val_list}, 跳过该 fold')
            continue
        train_data = build_dataset(train_list, voice_dict, emo_dict, pause_dict, energy_dict, tremor_dict, labels, args)
        val_data = build_dataset(val_list, voice_dict, emo_dict, pause_dict, energy_dict, tremor_dict, labels, args)
        torch.save(train_data, os.path.join(args.output_dir, f'data_train_{k}.pt'))
        torch.save(val_data, os.path.join(args.output_dir, f'data_val_{k}.pt'))
        print(f'[Fold {k}] 保存 data_train_{k}.pt / data_val_{k}.pt')
        if args.test_prefix:
            test_list = f'{args.test_prefix}{k}.scp'
            if os.path.isfile(test_list):
                test_data = build_dataset(test_list, voice_dict, emo_dict, pause_dict, energy_dict, tremor_dict, labels, args)
                torch.save(test_data, os.path.join(args.output_dir, f'data_test_{k}.pt'))
                print(f'[Fold {k}] 保存 data_test_{k}.pt')
            else:
                print(f'[Fold {k}] 未找到 {test_list} (跳过 test)')

    print('[DONE] 所有 fold 处理完成。')


if __name__ == '__main__':
    main()


