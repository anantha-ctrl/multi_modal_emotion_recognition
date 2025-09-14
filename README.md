# 🎭 Multi-modal Emotion Recognition System

## 🚀 Project Overview

This project is an advanced **Multi-modal Emotion Recognition System** that intelligently fuses video, audio, and text inputs to predict human emotions, including **Happy**, **Sad**, **Angry**, and **Neutral**. It’s built with state-of-the-art technologies like **PyTorch**, **Transformers**, **OpenCV**, and **Librosa**, wrapped in a simple **Streamlit** interface for easy interaction.

---

## ✅ Features

- ✅ Video emotion processing using facial keypoints
- ✅ Audio emotion extraction with MFCC features
- ✅ Text emotion analysis using BERT embeddings
- ✅ Multi-modal fusion model combining all three inputs
- ✅ Real-time interactive prediction web app
- ✅ Configurable via a clean `config.yaml` file
- ✅ Easily extendable for custom datasets

---

## 📸 Screenshot

Here’s what the app looks like in action:

!(https://res.cloudinary.com/de1tywvqm/image/upload/v1757834533/Screenshot_2025-09-14_123705_vcvnhm.png)

---

## ⚙️ Technologies Used

| Feature             | Technology / Library           |
|---------------------|--------------------------------|
| Video Processing    | OpenCV                         |
| Audio Processing    | Librosa                        |
| Text Processing     | HuggingFace Transformers (BERT) |
| Deep Learning Model | PyTorch                        |
| UI / Deployment     | Streamlit                      |
| Config Management   | PyYAML                         |

---

## 🏗️ Project Structure

```plaintext
multi_modal_emotion_recognition/
│
├── config.yaml
├── data/
├── models/
├── utils/
├── app/
├── train.py
├── requirements.txt
├── README.md
````

---

## ⚡ Setup Instructions

1️⃣ Clone the project:

```bash
git clone <your-repo-url>
cd multi_modal_emotion_recognition
```

2️⃣ Create and activate a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# OR
source .venv/bin/activate  # macOS/Linux
```

3️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

4️⃣ Run the Streamlit app:

```bash
streamlit run multi_modal_emotion_recognition/app/streamlit_app.py
```

---

## 🚀 How to Use

1. Open browser at `http://localhost:8501`.
2. Upload a sample video (.mp4), audio (.wav), and input some text.
3. Click **Predict Emotion** to see the predicted emotion and confidence.

---

## 📝 Sample Config (`config.yaml`)

```yaml
paths:
  video_sample: "./data/video/sample_video.mp4"
  audio_sample: "./data/audio/sample_audio.wav"
  text_sample: "./data/text/sample_text.txt"

parameters:
  frame_rate: 1
  text_max_length: 128
  num_classes: 4
```

---

## ⚠️ Notes

* Currently uses dummy random tensors in inference until real training and weight saving are implemented.
* Ensure proper file paths are configured in `config.yaml`.
* Trained model weight saving and loading will be added soon.

---

## 📚 Future Improvements

* Full dataset preprocessing pipeline
* Real training loop for fusion model
* Model weight-saving/loading system
* Enhanced visualization of prediction probabilities
* Docker support for deployment

---

## 💡 License

MIT License

---

Made with ❤️ by Anantha Kumar G

```

```
