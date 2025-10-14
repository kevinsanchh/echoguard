from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import tempfile # For creating temporary files
from pathlib import Path
import subprocess
import json # ADD THIS LINE
import base64 # AD

# ML Model Imports & Configuration - Copied from model.py
import torch
import torch.nn.functional as F
import torchaudio
import pandas as pd
import sys, warnings

# CONFIG (ensure these paths are correct relative to server.py)

MODEL_PATH = Path("./audio_resnet_model_best.pth")
CSV_PATH = Path("./dataset_train.csv")
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
global_clips_store = {}

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
        # Optionally, raise an exception or handle this more robustly
        sys.exit(1) # Exit if critical asset is missing

    df = pd.read_csv(CSV_PATH)
    global_classes = sorted(df['category'].unique())
    print(f"DEBUG: Loaded classes: {global_classes}")

    # Load Model
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        sys.exit(1) # Exit if critical asset is missing

    global_model = ResNetAudio(num_classes=len(global_classes)).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        global_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        global_model.load_state_dict(checkpoint)

    global_model.eval()
    print(f"DEBUG: ML Model loaded successfully on {DEVICE}.")

# Function to check if ffmpeg is available
def check_ffmpeg_presence():
    try:
        # Try to run a simple ffmpeg command to check its presence
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, text=True)
        print("INFO: 'ffmpeg' executable found and accessible on PATH.")
    except FileNotFoundError:
        print("CRITICAL ERROR: 'ffmpeg' executable NOT FOUND on system PATH. Please install FFmpeg.")
        print("  - For macOS: brew install ffmpeg")
        print("  - For Ubuntu/Debian: sudo apt install ffmpeg")
        print("  - For Windows: Download from ffmpeg.org and add to PATH.")
        sys.exit(1) # Exit if ffmpeg is critical and not found
    except subprocess.CalledProcessError as e:
        print(f"WARNING: 'ffmpeg' found but encountered an error running '-version': {e}")
        print(f"  Stderr: {e.stderr}")
    except Exception as e:
        print(f"WARNING: An unexpected error occurred while checking ffmpeg: {e}")

# Call this function to load ML assets when the app starts
with app.app_context():
    load_ml_assets()
    check_ffmpeg_presence() # ADD THIS CALL

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

    session_id = "default_session" # TODO: Implement proper session ID management from frontend

    webm_temp_path = None
    wav_temp_path = None

    try:
        incoming_suffix = '.webm'
        with tempfile.NamedTemporaryFile(delete=False, suffix=incoming_suffix, dir=app.instance_path) as tmp_file:
            audio_file.save(tmp_file.name)
            webm_temp_path = Path(tmp_file.name)
        
        if session_id not in global_clips_store:
            global_clips_store[session_id] = []
        global_clips_store[session_id].append(webm_temp_path)
        print(f"DEBUG: Stored clip {webm_temp_path} for session {session_id}. Total clips: {len(global_clips_store[session_id])}")

        print(f"DEBUG: Received audio file: {audio_file.filename}. Saved to {webm_temp_path}")
        print(f"DEBUG: File content type: {audio_file.content_type}")

        if global_model is None or not global_classes:
            raise RuntimeError("ML model or classes not loaded.")

        wav_suffix = '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=wav_suffix, dir=app.instance_path) as tmp_file_wav:
            wav_temp_path = Path(tmp_file_wav.name)

        ffmpeg_command = [
            'ffmpeg', '-y',
            '-i', str(webm_temp_path),
            str(wav_temp_path)
        ]
        
        print(f"DEBUG: Running FFmpeg command for immediate prediction: {' '.join(ffmpeg_command)}")
        result = subprocess.run( # Capture result to print stdout/stderr
            ffmpeg_command,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"FFmpeg stdout: {result.stdout}")
        if result.stderr:
            print(f"FFmpeg stderr: {result.stderr}")

        print(f"DEBUG: Successfully converted {webm_temp_path} to WAV at {wav_temp_path} for prediction.")

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
            'message': f'Audio file {audio_file.filename} received, converted, and processed for prediction!',
            'prediction': pred_class,
            'confidence': confidence
        }), 200

    except subprocess.CalledProcessError as e:
        print(f"ERROR: FFmpeg conversion failed (subprocess error): {e.stderr}")
        print(f"  FFmpeg stdout (if available): {e.stdout}")
        return jsonify({'error': f'FFmpeg conversion failed: {e.stderr}'}), 500
    except FileNotFoundError:
        print("CRITICAL ERROR: FFmpeg command not found. Please ensure FFmpeg is installed and on your system PATH.")
        return jsonify({'error': "FFmpeg command not found."}), 500
    except Exception as e:
        print(f"ERROR: Failed to process audio: {e}")
        return jsonify({'error': f'Failed to process audio: {str(e)}'}), 500
    finally:
        if wav_temp_path and wav_temp_path.exists():
            os.remove(wav_temp_path)
            print(f"DEBUG: Cleaned up temporary WAV file for prediction: {wav_temp_path}")

    print("ERROR: Fallback: Something went wrong processing the audio file (end of function).")
    return jsonify({'error': 'Something went wrong processing the audio file'}), 500   

@app.route("/api/stitch-and-return-audio", methods=['POST'])
def stitch_and_return_audio():
    session_id = "default_session" # TODO: Get session_id from request

    if session_id not in global_clips_store or not global_clips_store[session_id]:
        print(f"WARNING: No audio clips found for session {session_id} to stitch.")
        return jsonify({'error': 'No audio clips found for this session to stitch.'}), 404

    input_webm_files = global_clips_store[session_id]
    
    existing_files = [f for f in input_webm_files if f.exists()]
    if not existing_files:
        print(f"WARNING: No actual files found on disk for session {session_id}, despite paths in store.")
        del global_clips_store[session_id]
        return jsonify({'error': 'No actual audio files found on server to stitch.'}), 404
    
    # Sort files by creation/modification time if filenames are not sequential for robust concatenation
    # Assuming filenames are sequential due to Date.now() in frontend
    existing_files.sort() # Sorts by string representation, which should work for "audio_timestamp.webm"

    # Use a temporary directory for concatenation and final WAV
    with tempfile.TemporaryDirectory(dir=app.instance_path) as tmpdir_name:
        tmp_dir = Path(tmpdir_name)
        concat_list_path = tmp_dir / "concat_list.txt"
        stitched_webm_path = tmp_dir / "stitched_output.webm"
        final_wav_path = tmp_dir / "final_output.wav"

        with open(concat_list_path, "w") as f:
            for file_path in existing_files:
                f.write(f"file '{file_path.resolve()}'\n")

        print(f"DEBUG: Concatenating {len(existing_files)} WebM files for session {session_id} to {stitched_webm_path}...")
        print(f"DEBUG: Concat list content:\n{concat_list_path.read_text()}")

        try:
            # FFmpeg command to concatenate WebM files
            concat_command = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', str(concat_list_path),
                '-c', 'copy', str(stitched_webm_path)
            ]
            print(f"DEBUG: Running FFmpeg concat command: {' '.join(concat_command)}")
            result_concat = subprocess.run(
                concat_command,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"FFmpeg concat stdout: {result_concat.stdout}")
            if result_concat.stderr:
                print(f"FFmpeg concat stderr: {result_concat.stderr}")
            print(f"DEBUG: Successfully concatenated WebM files to {stitched_webm_path}")

            # Convert the stitched WebM to WAV
            convert_command = [
                'ffmpeg', '-y', '-i', str(stitched_webm_path), str(final_wav_path)
            ]
            print(f"DEBUG: Running FFmpeg convert command: {' '.join(convert_command)}")
            result_convert = subprocess.run(
                convert_command,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"FFmpeg convert stdout: {result_convert.stdout}")
            if result_convert.stderr:
                print(f"FFmpeg convert stderr: {result_convert.stderr}")
            print(f"DEBUG: Successfully converted stitched WebM to WAV at {final_wav_path}")

            with open(final_wav_path, "rb") as f:
                wav_data = f.read()
            
            encoded_wav = base64.b64encode(wav_data).decode('utf-8')
            data_uri = f"data:audio/wav;base64,{encoded_wav}"

            print(f"DEBUG: Generated Data URI for stitched audio. Size: {len(data_uri) / 1024:.2f} KB")

            return jsonify({'audioDataUri': data_uri}), 200

        except subprocess.CalledProcessError as e:
            print(f"ERROR: FFmpeg command failed (subprocess error) during stitching/conversion: {e.cmd}")
            print(f"  FFmpeg stdout (if available): {e.stdout}")
            print(f"  FFmpeg stderr (if available): {e.stderr}")
            return jsonify({'error': f'FFmpeg stitching/conversion failed: {e.stderr}'}), 500
        except FileNotFoundError:
            print("CRITICAL ERROR: FFmpeg command not found. Please ensure FFmpeg is installed and on your system PATH.")
            return jsonify({'error': "FFmpeg command not found."}), 500
        except Exception as e:
            print(f"ERROR: Failed to stitch and return audio: {e}")
            return jsonify({'error': f'Failed to stitch audio: {str(e)}'}), 500
        finally:
            for file_path in existing_files:
                if file_path.exists():
                    os.remove(file_path)
                    print(f"DEBUG: Cleaned up individual WebM clip: {file_path}")
            del global_clips_store[session_id]
            print(f"DEBUG: Cleaned up all temporary files and clips store for session {session_id}.")

    print(f"ERROR: Fallback: Something went wrong processing the stitching request (end of function).")
    return jsonify({'error': 'Something went wrong processing the stitching request'}), 500

# To run locally use 'python server.py'
if __name__ == "__main__":
    # Vercel sets the 'VERCEL' environment variable to '1' during deployment.
    # We check if this variable is NOT set before starting the development server.
    
    if os.environ.get('VERCEL') != '1':
        app.run(debug=True, port=8080)