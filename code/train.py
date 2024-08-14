#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :train.py
@Author :CodeCat
@Date   :2023/8/13 0:09
"""
import os
import math
import argparse
import time

import torch
import torchaudio
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model.audio_model import AudioNet
from data_utils.dataset import AudioDataset
from utils.train_val_one_epoch import train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    tb_writer = SummaryWriter('./logs')

    # 获取数据集
    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        win_length=args.n_fft // 2,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        normalized=True
    )
    data_transform = transforms.Compose([
            # transforms.Resize((args.n_mels, args.n_mels)),
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
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset=dataset,
        lengths=[num_train_dataset, num_val_dataset]
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw
    )
    # 获取模型
    model = AudioNet(
        pretrained=True,
        num_classes=args.num_classes,
        name=args.model_name
    ).to(device)

    model_weight_name = f'{args.model_name}_n_fft_{args.n_fft}_hop_length_{args.hop_length}_n_mels_{args.n_mels}'
    # 优化器
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5E-5)

    # cosine
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc = 0.0

    start = time.time()
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            device=device,
            epoch=epoch
        )

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_dataloader,
            device=device,
            epoch=epoch
        )

        print('[Epoch {}/{}] train_loss: {:.4f}, train_acc: {:.4f} || val_loss: {:.4f}, val_acc: {:.4f} || lr: {:.6f}'.format(
            epoch+1, args.epochs, train_loss, train_acc, val_loss, val_acc, optimizer.param_groups[0]['lr']
        ))
        # tensorboard
        tags = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'learning_rate']
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]['lr'], epoch)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./weights/" + model_weight_name +'_acc_'+ str(best_acc) + ".pth")
    end = time.time()
    print("Training 耗时为:{:.1f}".format(end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_rate', type=int, default=44100)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--hop_length', type=int, default=512)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--annotation_file', type=str, default='../UrbanSound8K/metadata/UrbanSound8K.csv')
    parser.add_argument('--audio_dir', type=str, default='../UrbanSound8K/audio')
    parser.add_argument('--num_samples', type=str, default=44100)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_name', type=str, default='mobilenetv3')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)

    opt = parser.parse_args()
    print(opt)
    main(opt)