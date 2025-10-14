from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import tempfile # For creating temporary files
from pathlib import Path

# ML Model Imports & Configuration - Copied from model.py
import torch
import torch.nn.functional as F
import torchaudio
import pandas as pd
import sys, warnings

# CONFIG (ensure these paths are correct relative to server.py)

MODEL_PATH = Path("./assets/audio_resnet_model_best.pth")
CSV_PATH = Path("./assets/dataset_train.csv")
SAMPLE_RATE = 16000
MAX_DURATION = 10 # This is the max duration the model expects, your frontend sends 5s snippets
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore")

# MODEL CLASSES - Copied from model.py
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_ch)
        self.conv2 = torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_ch)
        self.relu = torch.nn.ReLU(inplace=True)
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNetAudio(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = ResidualBlock(3, 32)
        self.layer2 = ResidualBlock(32, 64)
        self.layer3 = ResidualBlock(64, 128)
        self.layer4 = ResidualBlock(128, 128)
        self.pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.dropout = torch.nn.Dropout(0.15)
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# AUDIO PROCESSING FUNCTIONS - Copied from model.py
def load_audio(path):
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav.mean(dim=0, keepdim=True)  # mono
    max_length = SAMPLE_RATE * MAX_DURATION
    if wav.shape[1] > max_length:
        wav = wav[:, :max_length]
    else:
        wav = F.pad(wav, (0, max_length - wav.shape[1]))
    return wav

def preprocess(wav):
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
    )
    mel = transform(wav)
    mel_db = torchaudio.functional.amplitude_to_DB(mel, multiplier=10.0, amin=1e-10, db_multiplier=0.0)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    mel_db = mel_db.repeat(3, 1, 1)  # 3-channel input for ResNet
    return mel_db.unsqueeze(0)  # add batch dimension


# GLOBAL MODEL AND CLASSES
# These will be loaded once when the Flask app starts
global_model = None
global_classes = []

# app instance
app = Flask(__name__)
CORS(app)

if not os.path.exists(app.instance_path):
    os.makedirs(app.instance_path)
    print(f"DEBUG: Created instance folder: {app.instance_path}")

# Function to load the model and classes
def load_ml_assets():
    global global_model, global_classes

    # Load CSV to get class names
    if not CSV_PATH.exists():
        print(f"Error: CSV file not found at {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    global_classes = sorted(df['category'].unique())
    print(f"DEBUG: Loaded classes: {global_classes}")

    # Load Model
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        sys.exit(1)

    global_model = ResNetAudio(num_classes=len(global_classes)).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        global_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        global_model.load_state_dict(checkpoint)

    global_model.eval()
    print(f"DEBUG: ML Model loaded successfully on {DEVICE}.")

with app.app_context():
    load_ml_assets()
    # REMOVED: check_ffmpeg_presence() - not needed anymore

# Function to check if ffmpeg is available


# Call this function to load ML assets when the app starts
with app.app_context():
    load_ml_assets()

# Call this function to load ML assets when the app starts
with app.app_context():
    load_ml_assets()


# /api/home - Example endpoint
@app.route("/api/home", methods=['GET'])
def return_home():
    return jsonify({
        'message': "Example Message",
    })

# /api/audio-upload - Endpoint to receive and process audio
@app.route("/api/audio-upload", methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        print("ERROR: No 'audio' file part in the request.")
        return jsonify({'error': 'No audio file part in the request'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        print("ERROR: No selected file in the request.")
        return jsonify({'error': 'No selected file'}), 400

    wav_temp_path = None # Now we expect WAV directly

    try:
        # 1. Save the incoming audio file (which is now WAV from frontend ffmpeg.wasm)
        incoming_suffix = '.wav' # Expect WAV now
        with tempfile.NamedTemporaryFile(delete=False, suffix=incoming_suffix, dir=app.instance_path) as tmp_file:
            audio_file.save(tmp_file.name)
            wav_temp_path = Path(tmp_file.name)
        
        print(f"DEBUG: Received WAV file from frontend: {audio_file.filename}. Saved to {wav_temp_path}")
        print(f"DEBUG: File content type: {audio_file.content_type}")

        if global_model is None or not global_classes:
            raise RuntimeError("ML model or classes not loaded.")

        # 2. Preprocess and predict using the WAV file
        wav = load_audio(wav_temp_path)
        mel = preprocess(wav).to(DEVICE)

        with torch.no_grad():
            outputs = global_model(mel)
            probs = F.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)

            pred_class = global_classes[pred_idx.item()]
            confidence = round(conf.item() * 100, 2)

            print(f"DEBUG: Predicted: {pred_class} ({confidence:.2f}% confidence)")

        return jsonify({
            'message': f'Audio file {audio_file.filename} received and processed for prediction!',
            'prediction': pred_class,
            'confidence': confidence
        }), 200

    except Exception as e:
        print(f"ERROR: Failed to process audio: {e}")
        return jsonify({'error': f'Failed to process audio: {str(e)}'}), 500
    finally:
        # Clean up the temporary WAV file
        if wav_temp_path and wav_temp_path.exists():
            os.remove(wav_temp_path)
            print(f"DEBUG: Cleaned up temporary WAV file: {wav_temp_path}")
    
    print("ERROR: Fallback: Something went wrong processing the audio file (end of function).")
    return jsonify({'error': 'Something went wrong processing the audio file'}), 500   



# To run locally use 'python server.py'
if __name__ == "__main__":
    # Vercel sets the 'VERCEL' environment variable to '1' during deployment.
    # We check if this variable is NOT set before starting the development server.
    
    if os.environ.get('VERCEL') != '1':
        app.run(debug=True, port=8080)