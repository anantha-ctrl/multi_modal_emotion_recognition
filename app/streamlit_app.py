import streamlit as st
import torch
import yaml
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.video_model import VideoModel
from models.audio_model import AudioModel
from models.text_model import TextModel
from models.fusion_model import FusionModel
from utils.preprocess_audio import preprocess_audio
from utils.preprocess_text import preprocess_text
from utils.preprocess_video import preprocess_video

# Load config
with open('./config.yaml') as f:
    config = yaml.safe_load(f)

st.title("🎭 Multi-modal Emotion Recognition System")

uploaded_video = st.file_uploader("Upload Video", type=['mp4'])
uploaded_audio = st.file_uploader("Upload Audio", type=['wav'])
text_input = st.text_area("Enter text describing emotion")

if st.button("Predict Emotion"):
    st.write("Processing inputs...")

    # Dummy preprocessing (replace with real preprocessing pipeline)
    video_feat = torch.randn(1, 128)
    audio_feat = torch.randn(1, 128)
    input_ids, attention_mask = preprocess_text(text_input)
    text_feat = torch.randn(1, 768)

    # Load models
    fusion_model = FusionModel()
    fusion_model.eval()  # Make sure to set to evaluation mode

    output = fusion_model(video_feat, audio_feat, text_feat)
    predicted_idx = torch.argmax(output).item()

    emotions = ['Happy', 'Sad', 'Angry', 'Neutral']
    st.success(f"Predicted Emotion: {emotions[predicted_idx]}")
