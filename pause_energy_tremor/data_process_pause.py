# @Time    : 13/1/25
# @Author  : Minghui Zhao
# @Affiliation  : Southeast university

import wave
import numpy as np
import os
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display

import soundfile as sf

from pydub import AudioSegment
# from pydub.silence import split_on_silence
from pydub import silence

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def mel_plot(mel_spect, sr):
    plt.ion()
    #画mel谱图
    librosa.display.specshow(mel_spect, sr=sr, x_axis='time', y_axis='mel')
    plt.ylabel('Mel Frequency')
    plt.xlabel('Time(s)')
    plt.title('Mel Spectrogram')
    plt.show()
    plt.pause(0.5)

def pause_time(path):
    data_time = {}
    sound = AudioSegment.from_wav(path)
    loudness = sound.dBFS
    data_time['duration_seconds'] = sound.duration_seconds
    chunks = silence.split_on_silence(sound,
                                      min_silence_len=400,
                                      silence_thresh=loudness * 1.3,
                                      keep_silence=400
                                      )
    data_time['num'] = len(chunks)-1
    sum_pause = 0
    for i in range(len(chunks)):
        print(len(chunks[i]))
        sum_pause = sum_pause + len(chunks[i])
    time_pause = len(sound) - sum_pause
    print('time_pause', time_pause)
    data_time['time_pause'] = time_pause
    data_time['time_pause/all'] = time_pause
    return time_pause

def calculate_variance(data):
    n = len(data)
    if n < 2:
        return 0
    mean = sum(data) / n
    deviations = [(x - mean) for x in data]
    squared_deviations = [(x - mean)**2 for x in data]
    variance = sum(squared_deviations) / n
    return variance

def pause_time_gnn(path):
    data_time = {}
    sound = AudioSegment.from_wav(path)
    loudness = sound.dBFS
    print('time', sound.duration_seconds)
    silence_list = silence.detect_silence(sound, silence_thresh=loudness * 1.3, min_silence_len=200)
    print('silent', silence_list)
    data_time['duration_seconds'] = sound.duration_seconds*1000
    data_time['num'] = len(silence_list)
    if data_time['num'] != 0:
        sum_pause = 0
        vary_list = []
        for i,silence_chunk in enumerate(silence_list):
            i_silence_time = silence_chunk[1] - silence_chunk[0]
            vary_list.append(i_silence_time)
            sum_pause = sum_pause + i_silence_time
        data_time['time_vary'] = calculate_variance(vary_list)
        data_time['time_pause'] = sum_pause
        if data_time['duration_seconds'] == 0:
            data_time['time_pause/all'] = 0
        else:
            data_time['time_pause/all'] = sum_pause/data_time['duration_seconds']
        if sum_pause == 0:
            data_time['time_pause/speak'] = 0
        else:
            if (data_time['duration_seconds']-sum_pause) == 0:
                data_time['time_pause/speak'] = 0
            else:
                data_time['time_pause/speak'] = sum_pause / (data_time['duration_seconds'] - sum_pause)
    else:
        data_time['time_vary'] = 0
        data_time['time_pause'] = 0
        data_time['time_pause/all'] = 0
        data_time['time_pause/speak'] = 0
    return data_time

def int_sort(elem):
    return int(elem)

def wav_sort(elem):
    if elem.endswith(".wav"):
        wav_name = elem.split(".")[0]
    return int(wav_name)

def extract_features(root_dir, output_dir='pause'):
    """
    提取停顿相关特征。
    兼容两种目录结构：
      1) root_dir 下直接是大量 .wav
      2) root_dir 下是若干子目录，每个子目录里放若干 .wav

    参数:
        root_dir (str): wav 文件根目录
        output_dir (str): 特征输出目录（会自动创建）
    生成:
        每个音频一个 txt，6 维特征：
          0 duration_ms, 1 num_silence, 2 silence_duration_variance,
          3 total_silence_ms, 4 silence/total, 5 silence/speech
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"根目录不存在: {root_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # 收集所有 wav 路径
    all_wavs = []
    first_level = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
    for item in first_level:
        if os.path.isdir(item):
            # 子目录模式
            for sub in os.listdir(item):
                if sub.lower().endswith('.wav'):
                    all_wavs.append(os.path.join(item, sub))
        else:
            # 直接是文件
            if item.lower().endswith('.wav'):
                all_wavs.append(item)

    if len(all_wavs) == 0:
        print(f"未找到任何 wav 文件: {root_dir}")
        return

    print(f"共发现 {len(all_wavs)} 个 wav 文件，开始提取停顿特征...")

    for idx, wav_path in enumerate(sorted(all_wavs)):
        try:
            name_wav = os.path.splitext(os.path.basename(wav_path))[0]
            out_txt = os.path.join(output_dir, f"{name_wav}.txt")
            print(f"[{idx+1}/{len(all_wavs)}] {wav_path}")
            pause_feat = pause_time_gnn(wav_path)
            data_pause_mul = np.zeros((6), dtype=float)
            data_pause_mul[0] = pause_feat['duration_seconds']         # ms
            data_pause_mul[1] = pause_feat['num']                      # 段数
            data_pause_mul[2] = pause_feat['time_vary']                # 方差
            data_pause_mul[3] = pause_feat['time_pause']               # 静音总时长
            data_pause_mul[4] = pause_feat['time_pause/all']           # 静音/总时长
            data_pause_mul[5] = pause_feat['time_pause/speak']         # 静音/语音
            np.savetxt(out_txt, data_pause_mul, fmt='%.6f')
        except Exception as e:
            print(f"处理出错，跳过 {wav_path}: {e}")
            continue

    print('完成: 特征已保存到目录 ->', output_dir)



if  __name__== "__main__":
    # 可根据需要修改 root_dir
    extract_features('wav_files/save_wav_files_train_46_18', output_dir='pause')
