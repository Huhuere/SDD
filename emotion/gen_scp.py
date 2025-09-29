"""Utility to generate .scp file lists from one or more audio directories.

Example:
  python gen_scp.py \
    --dirs save_wav_files_train_46_18 save_wav_files_dev_20_20 save_wav_files_test_20_20 \
    --out-names train_0.scp val_0.scp test_0.scp \
    --ext .wav
"""

import argparse
import os
from pathlib import Path
from typing import List

def collect(dir_path: Path, ext: str) -> List[str]:
    files = []
    for p in dir_path.rglob('*'):
        if p.is_file() and p.suffix.lower() == ext.lower():
            # store as relative path from the emotion directory parent (current working dir when running)
            files.append(str(p.as_posix()))
    files.sort()
    return files

def make_relative(file_list: List[str], base_dir: Path) -> List[str]:
    rels = []
    for f in file_list:
        rels.append(str(Path(f).relative_to(base_dir).as_posix()))
    return rels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', nargs='+', required=True, help='One or more directories containing audio files')
    parser.add_argument('--out-names', nargs='+', required=True, help='Output .scp filenames (match length of --dirs)')
    parser.add_argument('--ext', default='.wav', help='Audio file extension to include (default .wav)')
    parser.add_argument('--lists-dir', default='lists', help='Directory to write the .scp files into')
    parser.add_argument('--strip-prefix', default=None, help='If provided, strip this leading path from each entry')
    args = parser.parse_args()

    if len(args.dirs) != len(args.out_names):
        raise ValueError('len(--dirs) must equal len(--out-names)')

    lists_dir = Path(args.lists_dir)
    lists_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path.cwd() if args.strip_prefix is None else Path(args.strip_prefix)

    for src_dir, out_name in zip(args.dirs, args.out_names):
        src_path = Path(src_dir)
        if not src_path.exists():
            print(f'[WARN] directory not found: {src_path}, skipping.')
            continue
        all_files = collect(src_path, args.ext)
        if args.strip_prefix is not None:
            rel_files = make_relative(all_files, base_dir)
        else:
            # ensure relative paths (avoid absolute so dataloader uses ./ prefix reliably)
            rel_files = [os.path.relpath(f, Path.cwd()).replace('\\', '/') for f in all_files]

        out_path = lists_dir / out_name
        with open(out_path, 'w', encoding='utf-8') as fw:
            for line in rel_files:
                fw.write(line + '\n')
        print(f'[INFO] Wrote {len(rel_files)} entries to {out_path}')

    print('[DONE] All requested .scp files generated.')

if __name__ == '__main__':
    main()

