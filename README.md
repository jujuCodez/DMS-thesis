# Driver Monitoring System (DMS) – Thesis Project 

A real-time AI-powered application designed to monitor driver behavior and alert for states such as **drowsiness**, **distraction**, or **mobile phone usage** using a lightweight CNN model and a conversational safety assistant powered by Gemini AI. Built for intelligent fleet monitoring and road safety enhancement.

---

##  Project Objectives

- Detect unsafe driver states from live camera feed
- Provide real-time visual and auditory feedback
- Integrate a conversational AI assistant for short verbal recommendations
- Enable lightweight, portable deployment using TorchScript
- Demonstrate multimodal driver monitoring on embedded-compatible interfaces

---

## Core Features

- 🎥 Live driver face detection and state classification (6 classes)
- 🧪 CNN inference using TorchScript (.pt models)
- 💬 Conversational feedback via Gemini AI
- 🔊 Audio alerts using TTS
- 📱 GUI developed with Kivy and KivyMD
- 🔄 Easily extendable to embedded platforms (Raspberry Pi, Jetson Nano, etc.)

---

## Tech Stack

| Layer        | Tools Used                          |
|--------------|-------------------------------------|
| Frontend     | Kivy, KivyMD                        |
| Backend      | Python, OpenCV, PyTorch             |
| AI Models    | MobileNetV3, MobileViT, EfficientNet (via timm) |
| Deployment   | TorchScript                         |
| Voice        | Plyer TTS, Gemini (Google Generative AI) |
| Data Format  | Preprocessed cropped face images    |

---

## Model Variants Trained

- `mobilenetv3_large_100`
- `mobilenetv3_small_100`
- `mobilevit_xs`
- `mobilevitv2_100`
- `efficientnet_lite0`
- `efficientnetv2_rw_t`

Each trained using TIMM + PyTorch, then converted to TorchScript via `pytorch_2_torchscript.py`.

---

## 🔧 Local Setup

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/your-repo/dms-thesis.git
cd dms-thesis
pip install -r requirements.txt
```

### 2. Create `.env` File

```bash
echo "GEMINI_API_KEY=your_key_here" > .env
```

### 3. Run the App

```bash
python app.py
```

---

## Usage

- The app captures video from a stream (`http://192.168.4.1:81/stream` by default)
- Predicts states like:
  - `focused`, `drowsy`, `sleepy`, `using phone`, `holding phone`, `yawning`
- Displays:
  - Frame with annotated face box and prediction
  - Cumulative counters for driver states
- Gemini AI triggers a short, context-aware voice reply when risky behavior is detected

---

## File Structure

| File / Folder                     | Description |
|----------------------------------|-------------|
| `app.py`                         | Main Kivy app and AI logic |
| `scenes.kv`                      | UI layout (screens) |
| `pytorch_2_torchscript.py`       | Converts .pth to TorchScript .pt |
| `model_trainer_notebook_*.ipynb` | Model training notebooks |
| `converted_model.pt`             | Pretrained model for inference |
| `.env`                           | Secret environment variables (not committed) |

---

## License

MIT License – Free for research and educational use.  
Commercial usage requires permission from the authors.

---

## Acknowledgments

Developed as part of a Computer Engineering undergraduate thesis project.  
Model architectures powered by the [TIMM library](https://github.com/huggingface/pytorch-image-models).

