import torch
import torch.nn as nn

class AudioModel(nn.Module):
    def __init__(self):
        super(AudioModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(40, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):  # x shape: (batch, time_steps, 40)
        x = x.permute(0, 2, 1)  # (batch, 40, time_steps)
        features = self.cnn(x).squeeze(-1)
        return features
