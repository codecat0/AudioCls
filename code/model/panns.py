#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :panns.py
@Author :CodeCat
@Date   :2023/8/19 22:24
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CNN6', 'CNN10', 'CNN14', 'cnn6', 'cnn10', 'cnn14']


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'max+avg':
            x = F.max_pool2d(x, kernel_size=pool_size) + F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception(
                f'Pooling type of {pool_type} is not supported. It must be one of "max", "avg" and "avg+max".'
            )
        return x


class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        x = self.con(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'max+avg':
            x = F.max_pool2d(x, kernel_size=pool_size) + F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception(
                f'Pooling type of {pool_type} is not supported. It must be one of "max", "avg" and "avg+max".'
            )
        return x


class CNN14(nn.Module):
    """
       The CNN14(14-layer CNNs) mainly consist of 6 convolutional blocks while each convolutional
       block consists of 2 convolutional layers with a kernel size of 3 × 3.

       Reference:
           PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
           https://arxiv.org/pdf/1912.10211.pdf
       """
    emb_size = 2048

    def __init__(self, num_melbins, extract_embedding=True):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(num_melbins)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc = nn.Linear(2048, self.emb_size)
        self.fc_audioset = nn.Linear(self.emb_size, 527)
        self.extact_embedding = extract_embedding

    def forward(self, x):
        # [bacth_size, 1, num_melbins, num_frame] -> [bacth_size, num_melbins, 1, num_frame]
        x = x.transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1, 2)

        x = self.conv_block1(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block3(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block4(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block5(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block6(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc(x))

        if self.extact_embedding:
            output = F.dropout(x, p=0.5, training=self.training)
        else:
            output = F.sigmoid(self.fc_audioset(x))
        return output


class CNN10(nn.Module):
    """
        The CNN10(14-layer CNNs) mainly consist of 4 convolutional blocks while each convolutional
        block consists of 2 convolutional layers with a kernel size of 3 × 3.

        Reference:
            PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
            https://arxiv.org/pdf/1912.10211.pdf
    """
    emb_size = 512

    def __init__(self, num_melbins, extract_embedding=True):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(num_melbins)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc = nn.Linear(512, self.emb_size)
        self.fc_audioset = nn.Linear(self.emb_size, 527)
        self.extract_embedding = extract_embedding

    def forward(self, x):
        # [bacth_size, 1, num_melbins, num_frame] -> [bacth_size, num_melbins, 1, num_frame]
        x = x.transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1, 2)

        x = self.conv_block1(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block3(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block4(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.mean(dim=3)
        x = x.mean(dim=2) + x.max(dim=2)[0]
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc(x), inplace=True)

        if self.extract_embedding:
            output = F.dropout(x, p=0.5, training=self.training)
        else:
            output = F.sigmoid(self.fc_audioset(x))

        return output


class CNN6(nn.Module):
    """
       The CNN14(14-layer CNNs) mainly consist of 4 convolutional blocks while each convolutional
       block consists of 1 convolutional layers with a kernel size of 5 × 5.

       Reference:
           PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
           https://arxiv.org/pdf/1912.10211.pdf
    """
    emb_size = 512

    def __init__(self, num_melbins, extract_embedding=True):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(num_melbins)
        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)

        self.fc = nn.Linear(512, self.emb_size)
        self.fc_audioset = nn.Linear(self.emb_size, 527)

        self.extract_embedding = extract_embedding

    def forward(self, x):
        # [bacth_size, 1, num_melbins, num_frame] -> [bacth_size, num_melbins, 1, num_frame]
        x = x.transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1, 2)

        x = self.conv_block1(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block3(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block4(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.mean(dim=3)
        x = x.mean(dim=2) + x.max(dim=2)[0]
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc(x), inplace=True)

        if self.extract_embedding:
            output = F.dropout(x, p=0.5, training=self.training)
        else:
            output = F.sigmoid(self.fc_audioset(x))
        return output


class SoundClassifier(nn.Module):
    def __init__(self, backbone, num_classes, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.backbone.emb_size, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def cnn14(num_melbins, extract_embedding=True, num_classes=10):
    backbone = CNN14(num_melbins=num_melbins, extract_embedding=extract_embedding)
    model = SoundClassifier(
        backbone=backbone,
        num_classes=num_classes
    )
    return model


def cnn10(num_melbins, extract_embedding=True, num_classes=10):
    backbone = CNN10(num_melbins=num_melbins, extract_embedding=extract_embedding)
    model = SoundClassifier(
        backbone=backbone,
        num_classes=num_classes
    )
    return model


def cnn6(num_melbins, extract_embedding=True, num_classes=10):
    backbone = CNN6(num_melbins=num_melbins, extract_embedding=extract_embedding)
    model = SoundClassifier(
        backbone=backbone,
        num_classes=num_classes
    )
    return model


if __name__ == '__main__':
    model = cnn10(num_melbins=128)
    input = torch.randn(size=(1, 1, 128, 74))
    output = model(input)
    print(output.shape)