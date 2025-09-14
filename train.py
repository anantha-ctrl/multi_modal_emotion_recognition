import torch
from torch.utils.data import DataLoader, Dataset
from models.video_model import VideoModel
from models.audio_model import AudioModel
from models.text_model import TextModel
from models.fusion_model import FusionModel
from transformers import AdamW
import torch.nn as nn

# Dummy Dataset Example (Replace with real data loader)
class EmotionDataset(Dataset):
    def __getitem__(self, idx):
        return {
            'video': torch.randn(5, 3, 224, 224),
            'audio': torch.randn(300, 40),
            'text_input_ids': torch.randint(0, 1000, (128,)),
            'text_attention_mask': torch.ones(128),
            'label': torch.tensor(0)
        }
    def __len__(self):
        return 100

dataset = EmotionDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

video_model = VideoModel().to(device)
audio_model = AudioModel().to(device)
text_model = TextModel().to(device)
fusion_model = FusionModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(list(video_model.parameters()) + list(audio_model.parameters()) + list(text_model.parameters()) + list(fusion_model.parameters()), lr=1e-4)

for epoch in range(10):
    for batch in dataloader:
        video = batch['video'].to(device)
        audio = batch['audio'].to(device)
        input_ids = batch['text_input_ids'].to(device)
        attention_mask = batch['text_attention_mask'].to(device)
        label = batch['label'].to(device)

        video_feat = video_model(video)
        audio_feat = audio_model(audio)
        text_feat = text_model(input_ids, attention_mask)

        output = fusion_model(video_feat, audio_feat, text_feat)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
