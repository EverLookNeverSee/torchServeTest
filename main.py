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


model = get_model()
layers = get_layers_to_fuse()
f = torch.quantization.fuse_modules(model, layers, inplace=False)
types_to_quantize = {torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU}
q = torch.quantization.quantize_dynamic(f, types_to_quantize, dtype=torch.qint8)
s = torch.jit.script(q)
torch.jit.save(s, "opt_model.pt")
