"""
    Main Script
    @Author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""

import torch
import torchvision


class ConvBnReluModel(torch.nn.Module):
    def __init__(self):
        super(ConvBnReluModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        self.bn = torch.nn.BatchNorm2d(5).to(dtype=torch.float)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def get_model():
    m = ConvBnReluModel()
    m.eval()
    return m


def get_layers_to_fuse():
    return [["conv", "bn", "relu"]]
