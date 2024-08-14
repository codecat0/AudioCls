#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :dataset.py
@Author :CodeCat
@Date   :2023/8/11 23:14
"""
import os
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms


class AudioDataset(Dataset):
    def __init__(self, annotation_file, audio_dir, mel_spectogram, transformation, target_sample_rate, num_samples, device):
        self.annotations = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.mel_spectogram = mel_spectogram
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        # self.device = device

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        # signal = signal.to(self.device)
        signal = self._resample_signal(signal, sr)
        signal = self._mix_down_signal(signal)
        signal = self._cut_signal(signal)
        signal = self._right_pad_signal(signal)
        signal = self.mel_spectogram(signal)
        signal = torch.divide(signal, torch.max(signal))
        signal = self.transformation(signal)
        return signal, label

    def _resample_signal(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.target_sample_rate
            )
            signal = resampler(signal)
        return signal

    def _cut_signal(self, signal):
        if signal.shape[1] > self.num_samples:
            start = random.randint(0, signal.shape[1] - self.num_samples)
            signal = signal[:, start : start + self.num_samples]
        return signal

    def _right_pad_signal(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (num_missing_samples // 2, num_missing_samples - num_missing_samples // 2)
            signal = F.pad(
                signal,
                last_dim_padding
            )
        return signal

    @staticmethod
    def _mix_down_signal(signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]

    def get_min_max_sample_rate(self):
        sample_rates = []
        for i in range(len(self.annotations)):
            sample_path = self._get_audio_sample_path(i)
            _, sr = torchaudio.load(sample_path)
            sample_rates.append(sr)
        return sample_rates


if __name__ == '__main__':
    ANNOTATION_FILE = '../../UrbanSound8K/metadata/UrbanSound8K.csv'
    AUDIO_DIR = '../../UrbanSound8K/audio'
    SAMPLE_RATE = 44100
    NUM_SAMPLES = 44100

    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        win_length=1024,
        hop_length=512,
        n_mels=128,
        normalized=True
    )
    transform = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=0.5, std=0.5)
        ]
    )

    urd = AudioDataset(
        annotation_file=ANNOTATION_FILE,
        audio_dir=AUDIO_DIR,
        mel_spectogram=mel_spectogram,
        transformation=transform,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        device='cpu'
    )

    # import matplotlib.pyplot as plt
    # for i in range(30, 130, 10):
    #     signal, label = urd[i]
    #     print(signal.shape, label)
    #     plt.imshow(signal[0].numpy(), aspect='auto')
    #     plt.title('class: ' + str(label))
    #     plt.colorbar()
    #     plt.show()
    # split_num = int(len(urd) * 0.8)
    # urd_train, urd_test = torch.utils.data.random_split(dataset=urd, lengths=[split_num, len(urd) - split_num])
    # print(len(urd_train))
    # print(len(urd_test))
    # signal, label = urd[0]
    # print(signal.shape)
    # print(torch.min(signal), torch.max(signal))
    for i in range(10, 100, 10):
        signal, label = urd[i]
        print(signal.shape)