# import torch.nn as nn
#
# class FusionModel(nn.Module):
#     def __init__(self):
#         super(FusionModel, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(128 + 128 + 768, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 4),  # 4 emotion classes
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, video_feat, audio_feat, text_feat):
#         fused = torch.cat([video_feat, audio_feat, text_feat], dim=1)
#         return self.fc(fused)

import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, input_dim=128 + 128 + 768, num_classes=4):
        super(FusionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, video_feat, audio_feat, text_feat):
        # Concatenate feature tensors along the feature dimension
        fused = torch.cat([video_feat, audio_feat, text_feat], dim=1)
        return self.fc(fused)
