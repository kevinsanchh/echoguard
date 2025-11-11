from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import tempfile # For creating temporary files
from pathlib import Path

# ML Model Imports & Configuration - Copied from model.py
import torch
import torch.nn.functional as F

from utils.config import MODEL_PATH, CSV_PATH, SAMPLE_RATE, MAX_DURATION, DEVICE
from utils.model_loader import load_ml_assets
from utils.audio_utils import load_audio, preprocess

# app instance
app = Flask(__name__)
CORS(app)

if not os.path.exists(app.instance_path):
    os.makedirs(app.instance_path)
    print(f"DEBUG: Created instance folder: {app.instance_path}")

# Load ML model and classes at startup once
# GLOBAL MODEL AND CLASSES
global_model, global_classes = load_ml_assets()
print("DEBUG: ML model and classes loaded successfully.")


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


# To run locally use 'python server.py'
if __name__ == "__main__":
    # Vercel sets the 'VERCEL' environment variable to '1' during deployment.
    # We check if this variable is NOT set before starting the development server.
    
    if os.environ.get('VERCEL') != '1':
        app.run(debug=True, port=8080)