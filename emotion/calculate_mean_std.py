import os
import math
import json
import argparse
import torch
import torchaudio
import numpy as np
import subprocess
import sys
try:
    import librosa
    _HAS_LIBROSA = True
except Exception:  # pragma: no cover
    _HAS_LIBROSA = False
try:
    import soundfile as sf  # pip install soundfile
    _HAS_SF = True
except Exception:  # pragma: no cover
    _HAS_SF = False
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='包含 wav 的目录 (可相对 / 绝对路径)')
    ap.add_argument('--ext', default='.wav', help='音频扩展名, 默认 .wav')
    ap.add_argument('--melbins', type=int, default=128, help='Mel 滤波器个数 (需与训练一致)')
    ap.add_argument('--target-length', type=int, default=512, help='帧长对齐 (需与训练一致)')
    ap.add_argument('--resample', type=int, default=None, help='可选：统一重采样到该采样率 (e.g. 16000)')
    ap.add_argument('--every-n', type=int, default=1, help='抽样间隔，仅处理每 N 个文件以加速')
    ap.add_argument('--stop-after', type=int, default=None, help='仅处理前 N 个文件做调试 (不含失败文件)')
    ap.add_argument('--verbose', action='store_true', help='打印每个文件处理信息')
    ap.add_argument('--log-errors', default=None, help='将失败文件和异常写入该日志文件')
    ap.add_argument('--fallback-mel', action='store_true', help='kaldi fbank 失败时使用 torchaudio MelSpectrogram 退化计算')
    ap.add_argument('--engine', choices=['kaldi','mel','librosa'], default='kaldi', help='特征计算后端：kaldi(默认) / mel(torchaudio MelSpectrogram) / librosa')
    ap.add_argument('--min-samples', type=int, default=400, help='过滤过短音频 (采样点数 < 此值)')
    ap.add_argument('--debug-files', action='store_true', help='逐文件打印 debug (即使没有 --verbose)')
    ap.add_argument('--no-progress', action='store_true', help='关闭 tqdm 进度条，避免某些终端刷新异常')
    ap.add_argument('--hard-exit-after-first', action='store_true', help='调试：成功处理第一个文件后立刻退出（用于分离崩溃是否与后续文件相关）')
    ap.add_argument('--load-backend', choices=['torchaudio','librosa','soundfile','wave'], default='torchaudio', help='音频读取后端 (与特征 engine 分离)')
    ap.add_argument('--isolate', action='store_true', help='隔离模式：每个文件用子进程读取，定位 native 崩溃文件 (较慢)')
    return ap.parse_args()


def load_audio(path, resample, backend='torchaudio'):
    """Load audio using selected backend. Return (1, N) tensor, sample_rate."""
    if backend == 'torchaudio':
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
    elif backend == 'librosa':
        if not _HAS_LIBROSA:
            raise RuntimeError('librosa 未安装')
        y, sr = librosa.load(path, sr=None, mono=True)
        if y.size == 0:
            raise RuntimeError('空音频')
        wav = torch.from_numpy(y).unsqueeze(0)
    elif backend == 'soundfile':
        if not _HAS_SF:
            raise RuntimeError('soundfile 未安装')
        y, sr = sf.read(path, always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if y.size == 0:
            raise RuntimeError('空音频')
        wav = torch.from_numpy(y).unsqueeze(0)
    elif backend == 'wave':
        import wave as _wave, numpy as _np
        with _wave.open(path, 'rb') as wf:
            sr = wf.getframerate()
            nchan = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
        if sampwidth == 2:
            dtype = np.int16
        elif sampwidth == 4:
            dtype = np.int32
        else:
            dtype = np.uint8
        y = np.frombuffer(raw, dtype=dtype)
        if nchan > 1:
            y = y.reshape(-1, nchan).mean(axis=1)
        if y.size == 0:
            raise RuntimeError('空音频')
        # 归一化到 -1~1
        if dtype == np.int16:
            y = y.astype(np.float32) / 32768.0
        elif dtype == np.int32:
            y = y.astype(np.float32) / 2147483648.0
        else:
            y = (y.astype(np.float32) - 128.0)/128.0
        wav = torch.from_numpy(y).unsqueeze(0)
    else:
        raise ValueError(f'未知 backend {backend}')

    if resample and sr != resample:
        wav = torchaudio.transforms.Resample(sr, resample)(wav)
        sr = resample
    if wav.numel() == 0:
        raise RuntimeError('空音频')
    wav = wav - wav.mean()
    return wav, sr


def wav_to_fbank_kaldi(wav, sr, melbins):
    return torchaudio.compliance.kaldi.fbank(
        wav,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type='hanning',
        num_mel_bins=melbins,
        dither=0.0,
        frame_shift=10,
    )

def wav_to_fbank_mel(wav, sr, melbins):
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        hop_length=int(sr * 0.01),
        win_length=int(sr * 0.025),
        window_fn=torch.hann_window,
        n_mels=melbins,
        center=True,
        power=2.0,
    )
    m = mel_tf(wav)  # [1, mel, frames]
    m = torch.log(m + 1e-6).squeeze(0).transpose(0, 1)
    return m

def wav_to_fbank_librosa(path, sr_target, melbins):
    if not _HAS_LIBROSA:
        raise RuntimeError('librosa 未安装，无法使用 engine=librosa')
    y, sr = librosa.load(path, sr=sr_target if sr_target else None, mono=True)
    if y.size == 0:
        raise RuntimeError('空音频(lr)')
    y = y - y.mean()
    n_fft = 1024
    hop = int((sr_target or sr) * 0.01)
    win = int((sr_target or sr) * 0.025)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win,
        window='hann', n_mels=melbins, power=2.0, center=True
    )  # [mel, frames]
    mel = np.log(mel + 1e-6).T  # [frames, mel]
    return torch.from_numpy(mel)


def pad_trim(fbank, target_length):
    p = target_length - fbank.shape[0]
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))
    elif p < 0:
        fbank = fbank[:target_length]
    return fbank


def main():
    args = parse_args()
    audio_dir = args.dir
    if not os.path.isdir(audio_dir):
        print(f'[ERROR] 目录不存在: {audio_dir}\n请确认路径是否正确 (当前工作目录: {os.getcwd()})')
        return 1

    files = [f for f in os.listdir(audio_dir) if f.lower().endswith(args.ext.lower())]
    files.sort()
    if len(files) == 0:
        print(f'[ERROR] 目录中没有匹配 {args.ext} 的文件: {audio_dir}')
        return 1

    if args.every_n > 1:
        files = files[::args.every_n]

    print(f'[INFO] 总文件数: {len(files)} (目录: {audio_dir}, 抽样间隔 every_n={args.every_n})')
    print('[INFO] 开始计算 (streaming sum / sumsq)...')

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    bad_files = 0

    processed_ok = 0
    error_log_fh = None
    if args.log_errors:
        error_log_fh = open(args.log_errors, 'w', encoding='utf-8')

    iterator = enumerate(files)
    if not args.no_progress:
        iterator = enumerate(tqdm(files, desc='Processing'))
    for idx, name in iterator:
        path = os.path.join(audio_dir, name)
        try:
            if args.debug_files:
                # 提前 flush，定位可能的 native crash 文件
                print(f'[DEBUG] 准备处理 {idx+1}/{len(files)}: {path}', flush=True)
            if args.engine == 'librosa':
                fb = wav_to_fbank_librosa(path, args.resample, args.melbins)
            else:
                if args.isolate:
                    # 子进程读取，避免当前进程被 native 崩溃拖死
                    code = ("import torch,sys,json;"
                            "import torchaudio,librosa,os;"
                            "path=sys.argv[1];res=int(sys.argv[2]) if sys.argv[2] != 'None' else None;backend=sys.argv[3];"
                            "import numpy as np;"
                            "import soundfile as sf if 'soundfile' in sys.modules else None")
                    # 这里简化，直接再次调用本脚本的 load 逻辑会很复杂，改为用 librosa 兜底
                    # 更简单：使用 librosa 加载 -> numpy -> torch
                    iso_cmd = [sys.executable, '-c', (
                        'import sys,torch;'
                        'try:'
                        ' import librosa; y,sr=librosa.load(sys.argv[1],sr=None,mono=True);'
                        ' import json;'
                        ' import numpy as np;'
                        ' import math;'
                        ' import os;'
                        ' import warnings;'
                        ' warnings.filterwarnings("ignore");'
                        ' import random;'
                        ' import time;'
                        ' import statistics;'
                        ' except Exception as e:'
                        '  print("ISO_FAIL:"+str(e)); sys.exit(2);'
                        ' import numpy as np;'
                        ' import math;'
                        ' import json;'
                        ' import sys;'
                        ' import os;'
                        ' import torch;'
                        ' t=torch.from_numpy(y).unsqueeze(0);'
                        ' t=t-t.mean();'
                        ' print(f"ISO_OK {t.shape[1]} {sr}")'
                    ), path]
                    try:
                        r = subprocess.run(iso_cmd, capture_output=True, text=True, timeout=20)
                        out = (r.stdout or '') + (r.stderr or '')
                        if r.returncode == 0 and out.startswith('ISO_OK'):
                            parts = out.strip().split()
                            # 伪造 wav, 重新用 librosa 再加载一次到主进程
                            wav, sr = load_audio(path, args.resample, backend='librosa')
                        else:
                            raise RuntimeError(f'隔离子进程失败: {out.strip()}')
                    except Exception as iso_e:
                        raise RuntimeError(f'隔离读取失败 {iso_e}')
                else:
                    wav, sr = load_audio(path, args.resample, backend=args.load_backend)
                if wav.shape[1] < args.min_samples:
                    raise RuntimeError(f'过短音频 samples={wav.shape[1]} < {args.min_samples}')
                try:
                    if args.engine == 'kaldi':
                        fb = wav_to_fbank_kaldi(wav, sr, args.melbins)
                    else:  # mel
                        fb = wav_to_fbank_mel(wav, sr, args.melbins)
                except Exception as inner_e:
                    if args.fallback_mel and args.engine == 'kaldi':
                        msg_fb = f'[FALLBACK] kaldi fbank 失败 -> mel {path} : {inner_e}'
                        if args.no_progress:
                            print(msg_fb)
                        else:
                            tqdm.write(msg_fb)
                        fb = wav_to_fbank_mel(wav, sr, args.melbins)
                    else:
                        raise inner_e
            fb = pad_trim(fb, args.target_length)
            fb64 = fb.to(torch.float64)
            total_sum += fb64.sum().item()
            total_sq_sum += (fb64 * fb64).sum().item()
            total_count += fb64.numel()
            processed_ok += 1
            if args.verbose and processed_ok <= 5:
                ok_msg = f'[OK] {path} frames={fb.shape[0]}'
                if args.no_progress:
                    print(ok_msg)
                else:
                    tqdm.write(ok_msg)
            if args.hard_exit_after_first and processed_ok == 1:
                print('[DEBUG] hard-exit-after-first 启用：已处理第 1 个文件，立即退出。')
                break
            if args.stop_after and processed_ok >= args.stop_after:
                if args.verbose or args.debug_files:
                    end_msg = f'[DEBUG] stop-after={args.stop_after} 达到, 提前结束'
                    if args.no_progress:
                        print(end_msg)
                    else:
                        tqdm.write(end_msg)
                break
        except Exception as e:
            bad_files += 1
            msg = f'[WARN] 读取失败 {path}: {e}'
            if args.no_progress:
                print(msg)
            else:
                tqdm.write(msg)
            if error_log_fh:
                error_log_fh.write(msg + '\n')
            continue

    if error_log_fh:
        error_log_fh.close()

    if total_count == 0:
        print('[ERROR] 没有成功处理任何文件。可能位置：')
        print('  1) 在第一个文件前 native 崩溃 (大概率 torchaudio 后端)')
        print('  2) 第一个文件损坏 / 编码不支持 导致底层崩溃')
        print('排查建议:')
        print('  a) 加 --debug-files 查看最后一条 [DEBUG] 输出的是哪个文件')
        print('  b) 改用 --engine librosa 试试 (纯 Python/FFmpeg 解码)')
        print('  c) 单独测试: python - <<EOF\nimport torchaudio;print(torchaudio.load(r"<某个文件>"))\nEOF')
        return 1

    mean = total_sum / total_count
    var = (total_sq_sum / total_count) - mean * mean
    var = max(var, 1e-12)
    std = math.sqrt(var)

    result = {
        'audio_dir': audio_dir,
        'files_used': len(files) - bad_files,
        'files_failed': bad_files,
        'melbins': args.melbins,
        'target_length': args.target_length,
        'resample': args.resample,
        'every_n': args.every_n,
        'mean': mean,
        'std': std,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n复制到提取脚本参数: --dataset-mean {mean:.6f} --dataset-std {std:.6f}")
    return 0


if __name__ == '__main__':

    raise SystemExit(main())
