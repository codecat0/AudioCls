#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :test.py
@Author :CodeCat
@Date   :2024/8/14 11:04
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来显示负号和中文的方法


def plot_wave_spec(wave_data):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("语谱图", fontsize=15)
    plt.specgram(wave_data, Fs=fs, scale_by_freq=True, sides='default', cmap="jet")
    plt.xlabel('秒/s', fontsize=15)
    plt.ylabel('频率/Hz', fontsize=15)

    plt.subplot(1, 2, 2)
    plt.title("波形图", fontsize=15)
    time = np.arange(0, len(wave_data)) * (1.0 / fs)
    plt.plot(time, wave_data)
    plt.xlabel('秒/s', fontsize=15)
    plt.ylabel('振幅', fontsize=15)

    plt.tight_layout()
    plt.show()


def get_spectrogram(x):
    D = librosa.stft(x, n_fft=2048)
    spect, phase = librosa.magphase(D)
    log_spect = np.log(spect)
    return spect, log_spect


def plot_spect(spectrogeam):
    plt.imshow(spectrogeam, aspect='auto', cmap='jet')
    plt.show()


fs = 173.16
txt_path = 'O/O001.txt'
data = []
with open(txt_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        data.append(int(line))

data = np.array(data).astype(np.float32)
plot_wave_spec(data)
