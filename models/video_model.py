import torch
import torch.nn as nn
from torchvision import models

class VideoModel(nn.Module):
    def __init__(self):
        super(VideoModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)

    def forward(self, x):  # x shape: (batch, frames, 3, 224, 224)
        batch_size, frames, C, H, W = x.shape
        x = x.view(batch_size * frames, C, H, W)
        features = self.resnet(x)
        features = features.view(batch_size, frames, -1)
        features = torch.mean(features, dim=1)  # Average across frames
        return features
