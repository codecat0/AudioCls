#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :audio_model.py
@Author :CodeCat
@Date   :2023/8/13 18:18
"""

import torch
import torch.nn as nn
from .alexnet import alexnet
from .vggnet import vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from .googlenet import googlenet
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .mobilenetv2 import MobileNetV2
from .mobilenetv3 import MobileNetV3, mobilenetv3_small, mobilenetv3_large
from .shufflenetv1 import shufflenet_g1, shufflenet_g2
from .shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
from .densenet import densenet121, densenet169, densenet201
from .ghostnet import GhostNet
from .vit import vit_base_patch16_224_in21k, vit_base_patch32_224_in21k


class AudioNet(nn.Module):
    def __init__(self, pretrained, num_classes=10, name='resnet'):
        super().__init__()
        self.input_layer = nn.Conv2d(1, 3, kernel_size=1)
        if name == 'alexnet':
            self.extract_feature = alexnet(pretrained=pretrained, num_classes=1000)
        elif name == 'vggnet':
            self.extract_feature = vgg16(pretrained=pretrained, num_classes=1000)
        elif name == 'googlenet':
            self.extract_feature = googlenet(pretrained=pretrained, num_classes=1000)
        elif name == 'resnet':
            self.extract_feature = resnet18(pretrained=pretrained, num_classes=1000)
        elif name == 'densenet':
            self.extract_feature = densenet121(pretrained=pretrained, num_classes=1000)
        elif name == 'mobilenetv2':
            self.extract_feature = MobileNetV2(num_classes=1000)
        elif name == 'mobilenetv3':
            self.extract_feature = mobilenetv3_large(num_classes=1000)
        elif name == 'shufflenetv1':
            self.extract_feature = shufflenet_g1(num_classes=1000)
        elif name == 'shufflenetv2':
            self.extract_feature = shufflenet_v2_x0_5(num_classes=1000)
        elif name == 'ghostnet':
            self.extract_feature = GhostNet()
        elif name == 'vit':
            self.extract_feature = vit_base_patch32_224_in21k(num_classes=1000)
        else:
            raise ValueError('Input extract feature model is not supported!!!')
        self.classifier_layer = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.extract_feature(x)
        x = self.classifier_layer(x)
        return x


if __name__ == '__main__':
    model = AudioNet(pretrained=False, num_classes=10)
    inputs = torch.randn(1, 1, 224, 224)
    out = model(inputs)
    print(out.shape)
