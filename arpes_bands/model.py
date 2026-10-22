"""The paper's valid-convolution 9-1-5 architecture (13 x 13 receptive field)."""

import torch
from torch import nn


class BandSRCNN(nn.Module):
    border = 6

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 9)
        self.conv2 = nn.Conv2d(64, 32, 1)
        self.conv3 = nn.Conv2d(32, 1, 5)

    def forward(self, intensity, return_features=False):
        if intensity.ndim != 4 or intensity.shape[1] != 1:
            raise ValueError("Expected [batch, 1, energy, momentum].")
        if min(intensity.shape[-2:]) < 13:
            raise ValueError("Both image dimensions must be at least 13.")
        f1 = torch.relu(self.conv1(intensity))
        f2 = torch.relu(self.conv2(f1))
        logits = self.conv3(f2)
        if return_features:
            return logits, {"layer1": f1, "layer2": f2}
        return logits

    def confidence(self, intensity):
        return self(intensity).sigmoid()
