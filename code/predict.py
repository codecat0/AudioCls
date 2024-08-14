#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :predict.py
@Author :CodeCat
@Date   :2023/8/13 18:04
"""
# !/usr/bin/env python
# -*-coding:utf-8 -*-

import os
import json
import argparse
import random

import pandas as pd
import torch
import torchaudio
from torchvision import transforms

from model.audio_model import AudioNet


def predict_signal(audio_path, model, mel_spectogram, data_transform, class_indict, real_label, device):
    assert os.path.exists(audio_path), f"file {audio_path} dose not exist."
    signal, sr = torchaudio.load(audio_path)
    if sr != args.sample_rate:
        signal = torchaudio.transforms.Resample(sr, args.sample_rate)(signal)
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    signal = mel_spectogram(signal)
    signal = torch.divide(signal, torch.max(signal))
    signal = data_transform(signal)

    signal = torch.unsqueeze(signal, dim=0)

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(signal.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "real: {}   predict: {}   prob: {:.3f}".format(real_label, class_indict[str(predict_cla)],
                                                               predict[predict_cla].numpy())
    print(print_res)
    return predict_cla


def main(args):
    random.seed(24)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        normalized=True
    )
    data_transform = transforms.Compose([
        transforms.Resize((args.n_mels, args.n_mels)),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    audio_path = args.audio_path

    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file {json_path} does not exist."
    json_file = open(json_path, 'r')
    class_indict = json.load(json_file)

    model = AudioNet(pretrained=True, num_classes=args.num_classes).to(device)
    print('Loading model state dict!!!')
    model.load_state_dict(torch.load(args.model_weight_path, map_location=device))
    print('Loaded model state dict!!!')
    annotations = pd.read_csv('../UrbanSound8K/metadata/UrbanSound8K.csv')

    random_select_audio_index = random.sample(range(0, annotations.shape[0]), 25)
    num_predict = 0
    num_predict_correct = 0
    for idx in random_select_audio_index:
        fold = f'fold{annotations.iloc[idx, 5]}'
        audio_path = os.path.join('../UrbanSound8K/audio', fold, annotations.iloc[idx, 0])
        real_label = annotations.iloc[idx, 7]
        real_class = annotations.iloc[idx, 6]
        predict_class = predict_signal(
            audio_path=audio_path,
            model=model,
            mel_spectogram=mel_spectogram,
            data_transform=data_transform,
            class_indict=class_indict,
            real_label=real_label,
            device=device
        )
        num_predict += 1
        if real_class == predict_class:
            num_predict_correct += 1
    print('Predict correct rate: {}'.format(num_predict_correct / num_predict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_rate', type=int, default=44100)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--hop_length', type=int, default=512)
    parser.add_argument('--n_mels', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--real_label', type=str, default='children_playing')
    parser.add_argument('--audio_path', type=str, default='../UrbanSound8K/audio/fold5/100263-2-0-117.wav')
    parser.add_argument('--model_weight_path', type=str, default='./weights/n_fft_1024_hop_length_512_n_mels_64_acc_0.8517458500286205.pth')

    args = parser.parse_args()
    main(args)