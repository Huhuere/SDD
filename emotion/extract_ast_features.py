"""Extract segment-level (and optional subject-level) AST emotion embeddings using a fine-tuned checkpoint.

Usage example (after you fine-tuned on RAVDESS and got a best checkpoint):

    python extract_ast_features.py \
        --scp daic_all.scp \
        --dataset daic \
        --checkpoint exp/fold0/models/best_audio_model0.85.pth \
        --output-dir ../features_emotion_daic \
        --batch-size 32 \
        --audio-length 512 \
        --dataset-mean -8.73210334777832 \
        --dataset-std 6.587666034698486
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
from typing import List, Dict, Tuple

basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)

import dataloader  # noqa: E402
import models      # noqa: E402


def infer_label_dim_from_checkpoint(sd: Dict[str, torch.Tensor]) -> int:
    """Try to infer the classifier output dim from saved state dict keys."""
    # Look for mlp_head.*.weight last Linear layer weight (shape [out_dim, in_dim])
    candidate = [k for k in sd.keys() if k.endswith('mlp_head.1.weight') or k.endswith('mlp_head.0.weight')]
    # In our ASTModel definition: self.mlp_head = LayerNorm + Linear => keys: mlp_head.0.weight, mlp_head.1.weight
    for k in candidate:
        weight = sd[k]
        if weight.ndim == 2:
            return weight.shape[0]
    # fallback: common defaults (ravdess 8-class or binary)
    if any('mlp_head' in k for k in sd.keys()):
        # Heuristic: if ravdess fine-tuned -> 8
        return 8
    return 2


def build_dataloader(args):
    audio_conf = {
        'num_mel_bins': 128,
        'target_length': args.audio_length,
        'freqm': 0,  # no augmentation for extraction
        'timem': 0,
        'mixup': 0,
        'dataset': args.dataset,
        'mode': 'evaluation',
        'mean': args.dataset_mean,
        'std': args.dataset_std,
        'noise': False,
        'fbank_engine': args.fbank_engine,
        'fbank_fallback': args.fbank_fallback,
        'allow_fail': True,
    }
    dataset = dataloader.AudiosetDataset(
        dataset_json_file=args.scp,
        dataset_name=args.dataset,
        label_csv='',
        audio_conf=audio_conf,
        audio_class=args.n_class
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    return loader


def fit_projection(emb: np.ndarray, out_dim: int, method: str = 'svd') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a linear projection (like PCA) using SVD.
    Returns: (projected_embeddings, mean_vec, proj_matrix)
    emb: (N, D)
    out_dim: target dim (< D)
    proj_matrix: shape (D, out_dim)
    """
    if method == 'none' or out_dim == 0 or out_dim >= emb.shape[1]:
        return emb, np.zeros((emb.shape[1],), dtype=np.float32), np.eye(emb.shape[1], dtype=np.float32)[:, :emb.shape[1]]
    # center
    mean = emb.mean(axis=0, keepdims=True)
    Xc = emb - mean
    # economy SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:out_dim].T  # (D, out_dim)
    Z = Xc @ W
    return Z.astype(np.float32), mean.squeeze(0).astype(np.float32), W.astype(np.float32)


def apply_projection(emb: np.ndarray, mean_vec: np.ndarray, proj_matrix: np.ndarray) -> np.ndarray:
    Xc = emb - mean_vec.reshape(1, -1)
    Z = Xc @ proj_matrix
    return Z.astype(np.float32)


def extract_embeddings(model, loader, device, args):
    model.eval()
    all_embeddings = []
    all_logits = []
    all_labels = []
    all_names = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                audio_input, labels, names = batch
            else:
                audio_input, labels = batch
                names = [f'sample_{len(all_names)+i}' for i in range(len(audio_input))]
            audio_input = audio_input.to(device)
            logits, embedding = model(audio_input)
            all_embeddings.append(embedding.cpu())
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_names.extend(names)
    embeddings = torch.cat(all_embeddings).numpy()
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return embeddings, logits, labels, all_names


def aggregate_by_subject(embeddings: np.ndarray, names: List[str], args):
    subj_to_vecs = {}
    subj_rule_idx = args.subject_split_idx
    for emb, nm in zip(embeddings, names):
        base = os.path.basename(nm)
        parts = base.split('_')
        if len(parts) <= subj_rule_idx:
            subj_id = parts[0]
        else:
            subj_id = parts[subj_rule_idx]
        subj_to_vecs.setdefault(subj_id, []).append(emb)
    subj_ids = []
    subj_feats = []
    for k, vs in subj_to_vecs.items():
        subj_ids.append(k)
        subj_feats.append(np.mean(vs, axis=0))
    return np.vstack(subj_feats), subj_ids


def main():
    parser = argparse.ArgumentParser(description="Extract AST emotion embeddings from a fine-tuned checkpoint")
    parser.add_argument('--scp', required=True, help='Path to .scp list (one filename per line)')
    parser.add_argument('--dataset', default='daic', help='Dataset keyword (affects path + label parsing)')
    parser.add_argument('--checkpoint', required=True, help='Path to fine-tuned AST checkpoint (.pth)')
    parser.add_argument('--output-dir', required=True, help='Directory to store extracted features')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--audio-length', type=int, default=512, help='#frames (time) used during fine-tuning')
    parser.add_argument('--dataset-mean', type=float, default=-8.73210334777832)
    parser.add_argument('--dataset-std', type=float, default=6.587666034698486)
    parser.add_argument('--n-class', type=int, default=8, help='Original label dim of fine-tuned model (for head)')
    parser.add_argument('--fstride', type=int, default=10)
    parser.add_argument('--tstride', type=int, default=10)
    parser.add_argument('--imagenet-pretrain', type=str, default='True')
    parser.add_argument('--audioset-pretrain', type=str, default='True')
    parser.add_argument('--subject-split-idx', type=int, default=0, help='Index in underscore split used as subject id')
    # Per-segment npy now always saved (flag retained for backward compatibility, ignored)
    parser.add_argument('--save-segment-npy', action='store_true', help='(Deprecated) Was used to toggle per-segment saving; now always saved.')
    parser.add_argument('--device', default='auto', help='cpu / cuda / auto')
    parser.add_argument('--no-subject-agg', action='store_true', help='Skip subject-level aggregation, only output segment-level features')
    # argparse converts --no-subject-agg to attribute no_subject_agg
    # re-added feature engine options (previous edit was overwritten)
    parser.add_argument('--fbank-engine', default='kaldi', choices=['kaldi','mel','librosa'], help='Feature extraction backend for fbank/log-mel')
    parser.add_argument('--fbank-fallback', default=None, help='Optional fallback engine if primary fails (e.g., mel or librosa)')
    # Projection arguments
    parser.add_argument('--proj-dim', type=int, default=256, help='Target embedding dim (set 0 or >=768 to disable)')
    parser.add_argument('--proj-method', choices=['svd','none'], default='svd', help='Projection method: svd (PCA-like) or none')
    parser.add_argument('--save-projection', action='store_true', help='Save fitted projection params (mean & matrix)')
    parser.add_argument('--apply-existing-projection', default=None, help='Directory containing projection_mean.npy and projection_matrix.npy to reuse')

    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    # Per-segment directory will always be created later (legacy flag retained but ignored)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in ckpt:  # in case saved via ddp style
        sd = ckpt['state_dict']
    else:
        sd = ckpt
    inferred_dim = infer_label_dim_from_checkpoint(sd)
    print(f"[INFO] Inferred classifier output dim from checkpoint: {inferred_dim}")

    # Instantiate model with that label_dim so weights load cleanly
    audio_model = models.ASTModel(
        label_dim=inferred_dim,
        fstride=args.fstride,
        tstride=args.tstride,
        input_fdim=128,
        input_tdim=args.audio_length,
        imagenet_pretrain=(args.imagenet_pretrain == 'True'),
        audioset_pretrain=(args.audioset_pretrain == 'True'),
        model_size='base384',
        verbose=True
    )
    missing, unexpected = audio_model.load_state_dict(sd, strict=False)
    print(f"[WARN] Missing keys: {len(missing)}; Unexpected keys: {len(unexpected)}")
    if missing:
        print("  Missing (first 10):", missing[:10])
    if unexpected:
        print("  Unexpected (first 10):", unexpected[:10])

    audio_model.to(device)

    # Build dataloader
    loader = build_dataloader(args)
    print(f"[INFO] Using fbank_engine={args.fbank_engine} fallback={args.fbank_fallback}")
    print(f"[INFO] Start embedding extraction on {len(loader.dataset)} segments ...")
    t0 = time.time()
    embeddings, logits, labels, names = extract_embeddings(audio_model, loader, device, args)
    dur = time.time() - t0
    print(f"[INFO] Extraction finished in {dur/60:.2f} min. Raw shape: {embeddings.shape}")

    # Projection handling
    original_dim = embeddings.shape[1]
    proj_used = False
    if args.apply_existing_projection:
        mean_p = os.path.join(args.apply_existing_projection, 'projection_mean.npy')
        mat_p = os.path.join(args.apply_existing_projection, 'projection_matrix.npy')
        if not (os.path.isfile(mean_p) and os.path.isfile(mat_p)):
            raise FileNotFoundError(f"[ERR] projection files not found in {args.apply_existing_projection}")
        mean_vec = np.load(mean_p)
        proj_matrix = np.load(mat_p)
        embeddings = apply_projection(embeddings, mean_vec, proj_matrix)
        proj_used = True
        print(f"[INFO] Applied existing projection: {original_dim} -> {embeddings.shape[1]}")
    else:
        if args.proj_dim and 0 < args.proj_dim < original_dim and args.proj_method != 'none':
            embeddings, mean_vec, proj_matrix = fit_projection(embeddings, args.proj_dim, args.proj_method)
            proj_used = True
            print(f"[INFO] Fitted projection {args.proj_method}: {original_dim} -> {embeddings.shape[1]}")
            if args.save_projection:
                np.save(os.path.join(args.output_dir, 'projection_mean.npy'), mean_vec)
                np.save(os.path.join(args.output_dir, 'projection_matrix.npy'), proj_matrix)
                print('[INFO] Saved projection_mean.npy & projection_matrix.npy')
        else:
            print('[INFO] Projection skipped (use --proj-dim <768 and --proj-method svd to enable).')

    # Save segment-level outputs
    np.save(os.path.join(args.output_dir, 'segment_features.npy'), embeddings)
    np.savetxt(os.path.join(args.output_dir, 'segment_filenames.txt'), names, fmt='%s')
    np.save(os.path.join(args.output_dir, 'segment_logits.npy'), logits)
    np.save(os.path.join(args.output_dir, 'segment_labels.npy'), labels)
    # Always save per-segment file after (optional) projection
    seg_dir = os.path.join(args.output_dir, 'segments')
    os.makedirs(seg_dir, exist_ok=True)
    for emb, nm in zip(embeddings, names):
        base = os.path.splitext(os.path.basename(nm))[0]
        np.save(os.path.join(seg_dir, base + '.npy'), emb)
    print(f'[INFO] Saved {len(names)} per-segment .npy files (dim={embeddings.shape[1]}).')

    if not args.no_subject_agg:
        subj_feats, subj_ids = aggregate_by_subject(embeddings, names, args)
        np.save(os.path.join(args.output_dir, 'subject_mean_features.npy'), subj_feats)
        np.savetxt(os.path.join(args.output_dir, 'subject_ids.txt'), subj_ids, fmt='%s')
        print(f"[INFO] Subject-level aggregation done: {subj_feats.shape}")
    else:
        print('[INFO] Skipped subject-level aggregation (--no-subject-agg).')
    if proj_used:
        print('[INFO] NOTE: Downstream fusion should set num_emotion_features to the projected dimension '
              f'({embeddings.shape[1]}).')
    print("[DONE] Emotion feature extraction pipeline complete.")


if __name__ == '__main__':
    main()

