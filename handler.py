"""
    Main Handler Script
    @Author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""


import io
import torch
from PIL import Image
from requests import request
import torch.nn.functional as F
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
