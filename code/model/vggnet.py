#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :vggnet.py
@Author :CodeCat
@Date   :2023/8/19 21:23
"""

import torch
import torch.nn as nn

from torch.hub import load_state_dict_from_url

__all__ = ['vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']

model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-8a719046.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-19584684.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True, dropout=0.5):
        super(VGG, self).__init__()
        self.features = features
        # 这一操作是为了保证特征提取后的特征图大小为 7x7，使得网络可以接受224x224以外尺寸的图像
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # 提取图像特征
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        # 实现图像分类
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """
        模型权重初始化
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)
            # 论文中没有batch_normaliztion，当时这个还没有提出
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    return _vgg(
        arch='vgg11',
        cfg='A',
        batch_norm=False,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    return _vgg(
        arch='vgg11_bn',
        cfg='A',
        batch_norm=True,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def vgg13(pretrained=False, progress=True, **kwargs):
    return _vgg(
        arch='vgg13',
        cfg='B',
        batch_norm=False,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    return _vgg(
        arch='vgg13_bn',
        cfg='B',
        batch_norm=True,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def vgg16(pretrained=False, progress=True, **kwargs):
    return _vgg(
        arch='vgg16',
        cfg='D',
        batch_norm=False,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    return _vgg(
        arch='vgg16_bn',
        cfg='D',
        batch_norm=True,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def vgg19(pretrained=False, progress=True, **kwargs):
    return _vgg(
        arch='vgg19',
        cfg='E',
        batch_norm=False,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    return _vgg(
        arch='vgg19_bn',
        cfg='E',
        batch_norm=True,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


if __name__ == '__main__':
    inputs = torch.randn(1, 3, 224, 224)
    model = vgg19(num_classes=10)
    out = model(inputs)
    print(out.shape)
