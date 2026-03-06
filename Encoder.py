import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class ResNetEncoder(nn.Module):
    """
    ResNet18에서 마지막 conv feature map (B,C,H',W')만 뽑는 Encoder
    """
    def __init__(self, pretrained: bool = True, out_layer: str = "layer4"):
        super().__init__()
        backbone = models.resnet18(pretrained=pretrained)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.out_layer = out_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        if self.out_layer == "layer1":
            return x
        x = self.layer2(x)
        if self.out_layer == "layer2":
            return x
        x = self.layer3(x)
        if self.out_layer == "layer3":
            return x
        x = self.layer4(x)
        return x
