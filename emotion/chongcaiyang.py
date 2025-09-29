import os, librosa, soundfile as sf

src_root = 'Audio_Speech_Actors_01-24'
dst_root = 'ravdess_ast_16k'
os.makedirs(dst_root, exist_ok=True)

for root, _, files in os.walk(src_root):
    for f in files:
        if f.lower().endswith('.wav'):
            full = os.path.join(root, f)
            y, sr = librosa.load(full, sr=16000)  # 重采样到16k
            sf.write(os.path.join(dst_root, f), y, 16000)