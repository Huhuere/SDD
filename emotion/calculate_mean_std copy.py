import torchaudio, torch, os
import numpy as np

melbins = 128
target_length = 512  # 与训练一致
all_vals = []

for name in os.listdir('ravdess_ast_16k'):
    if not name.endswith('.wav'): continue
    wav, sr = torchaudio.load(os.path.join('ravdess_ast_16k', name))
    wav = wav - wav.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        wav, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=melbins, dither=0.0, frame_shift=10
    )
    p = target_length - fbank.shape[0]
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0,0,0,p))
    elif p < 0:
        fbank = fbank[:target_length]
    all_vals.append(fbank)

stack = torch.cat(all_vals, dim=0)  # (总帧数, melbins)
mean = stack.mean().item()
std = stack.std().item()
print('RAVDESS mean:', mean, 'std:', std)