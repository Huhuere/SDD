"""Compute global Log-Mel (Kaldi fbank) mean and std for a dataset listed in an .scp file.

Why: For feature extraction (e.g., AST) you passed RAVDESS's mean/std. It's better to recompute
dataset-specific statistics (DAIC, etc.) to match distribution and improve convergence.

Workflow:
 1. Prepare an .scp file (each line: relative path to wav) e.g. lists/train_0.scp
 2. Ensure audio are mono 16 kHz (or consistent sample rate). If not 16k, the script will use
    whatever sr the file has (matching dataloader logic). If you need enforced 16k resampling,
    add --resample 16000.
 3. Run:
       python compute_fbank_stats.py \
           --scp lists/train_0.scp \
           --audio-length 512 \
           --num-mel-bins 128 \
           --resample 16000
 4. It prints mean / std you can plug into extract_ast_features.py

Details:
 - We compute fbank identical to dataloader (_wav2fbank): Kaldi fbank, hanning window, frame_shift=10ms
 - Zero-padding / truncation to target_length so stats align with training configuration
 - No SpecAugment, no mixup, no normalization inside

Output:
   Prints JSON summary and optionally saves stats to a .npz (--save-path)

Note: For multi-split consistency (train/dev/test) you can:
   (a) Use only train split stats (recommended)
   (b) Concatenate all .scp into one big list then run
"""

import argparse
import os
import json
import math
from pathlib import Path
from typing import List

import torch
import torchaudio
from tqdm import tqdm


def read_scp(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines


def load_audio(path: str, resample: int | None):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:  # convert to mono by mean
        wav = torch.mean(wav, dim=0, keepdim=True)
    if resample and sr != resample:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=resample)(wav)
        sr = resample
    wav = wav - wav.mean()
    return wav, sr


def wav_to_fbank(waveform: torch.Tensor, sr: int, num_mel_bins: int):
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type='hanning',
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_shift=10,
    )  # shape [frames, mel]
    return fbank


def pad_or_trim(fbank: torch.Tensor, target_length: int):
    n_frames = fbank.shape[0]
    diff = target_length - n_frames
    if diff > 0:
        pad = torch.nn.ZeroPad2d((0, 0, 0, diff))
        fbank = pad(fbank)
    elif diff < 0:
        fbank = fbank[:target_length]
    return fbank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scp', required=True, help='Path to .scp list (wav relative or absolute paths)')
    ap.add_argument('--audio-root', default='.', help='Optional root to prepend if .scp entries are relative')
    ap.add_argument('--audio-length', type=int, default=512, help='Target number of frames (time) to align with training')
    ap.add_argument('--num-mel-bins', type=int, default=128)
    ap.add_argument('--resample', type=int, default=None, help='Resample to this rate (e.g., 16000); if omitted use file sr')
    ap.add_argument('--save-path', default=None, help='If set, save stats npz here')
    ap.add_argument('--every-n', type=int, default=1, help='Subsample: only process every N-th file for quick estimate')
    args = ap.parse_args()

    files = read_scp(args.scp)
    if args.every_n > 1:
        files = files[::args.every_n]
    if len(files) == 0:
        raise RuntimeError('No entries in scp file.')

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for fp in tqdm(files, desc='Computing fbank stats'):
        full_path = fp if os.path.isabs(fp) else os.path.join(args.audio_root, fp)
        if not os.path.exists(full_path):
            tqdm.write(f'[WARN] missing file: {full_path}; skip')
            continue
        try:
            wav, sr = load_audio(full_path, args.resample)
            fbank = wav_to_fbank(wav, sr, args.num_mel_bins)
            fbank = pad_or_trim(fbank, args.audio_length)
            # accumulate stats (use double precision to reduce drift)
            fb64 = fbank.to(torch.float64)
            total_sum += fb64.sum().item()
            total_sq_sum += (fb64 * fb64).sum().item()
            total_count += fb64.numel()
        except Exception as e:
            tqdm.write(f'[ERR] {full_path}: {e}')
            continue

    if total_count == 0:
        raise RuntimeError('No valid frames accumulated. Check paths.')

    mean = total_sum / total_count
    var = (total_sq_sum / total_count) - mean * mean
    var = max(var, 1e-12)
    std = math.sqrt(var)

    stats = {
        'files_processed': len(files),
        'total_frames': total_count,
        'mean': mean,
        'std': std,
        'num_mel_bins': args.num_mel_bins,
        'target_length': args.audio_length,
        'resample': args.resample,
        'every_n': args.every_n,
    }
    print(json.dumps(stats, indent=2))

    if args.save_path:
        import numpy as np
        np.savez(args.save_path, **stats)
        print(f'[INFO] Saved stats to {args.save_path}')


if __name__ == '__main__':
    main()
