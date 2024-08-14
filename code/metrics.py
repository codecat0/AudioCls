#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :metrics.py
@Author :CodeCat
@Date   :2023/8/20 20:23
"""
import json
import os
import argparse

import torch
import torchaudio
from torch.utils.data import DataLoader
from torchvision import transforms

from model.audio_model import AudioNet
from data_utils.dataset import AudioDataset
from utils.multi_class_metrics import model_metrics, get_roc_pr, plot_roc, plot_pr


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])

    # 获取数据集
    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        # win_length=args.n_fft // 2,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        normalized=True
    )
    data_transform = transforms.Compose([
            transforms.Resize((args.n_mels, args.n_mels)),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
    dataset = AudioDataset(
        annotation_file=args.annotation_file,
        audio_dir=args.audio_dir,
        mel_spectogram=mel_spectogram,
        transformation=data_transform,
        target_sample_rate=args.sample_rate,
        num_samples=args.num_samples,
        device=device
    )
    num_train_dataset = int(len(dataset) * 0.8)
    num_val_dataset = len(dataset) - num_train_dataset
    _, val_dataset = torch.utils.data.random_split(
        dataset=dataset,
        lengths=[num_train_dataset, num_val_dataset]
    )
    dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw
    )
    # 获取模型
    model = AudioNet(
        pretrained=True,
        num_classes=args.num_classes,
        name=args.model_name
    ).to(device)
    print('Loading model state dict!!!')
    model.load_state_dict(torch.load(args.model_weight_path, map_location=device))
    print('Loaded model state dict!!!')

    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file {json_path} does not exist."
    json_file = open(json_path, 'r')
    class_indict = json.load(json_file)

    trues, preds, probs = model_metrics(
        model=model,
        dataloader=dataloader,
        device=device,
        class_indict=class_indict
    )

    fpr, tpr, auc, precision, recall, ap = get_roc_pr(
        trues=trues,
        preds=preds,
        probs=probs
    )

    plot_roc(
        fpr=fpr,
        tpr=tpr,
        roc_auc=auc,
        class_indict=class_indict
    )

    plot_pr(
        precision=precision,
        recall=recall,
        ap=ap,
        class_indict=class_indict
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_rate', type=int, default=44100)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--hop_length', type=int, default=512)
    parser.add_argument('--n_mels', type=int, default=64)
    parser.add_argument('--annotation_file', type=str, default='../UrbanSound8K/metadata/UrbanSound8K.csv')
    parser.add_argument('--audio_dir', type=str, default='../UrbanSound8K/audio')
    parser.add_argument('--num_samples', type=str, default=44100)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_name', type=str, default='resnet')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--model_weight_path', type=str, default='./weights/n_fft_1024_hop_length_512_n_mels_64_acc_0.8517458500286205.pth')

    opt = parser.parse_args()
    print(opt)
    main(opt)