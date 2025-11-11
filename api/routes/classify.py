import os
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from flask import Blueprint, jsonify, request, current_app
from utils.audio_utils import load_audio, preprocess

# -----------------------
# Blueprint setup
# -----------------------

classify_bp = Blueprint("classify", __name__, url_prefix="/api")

# -----------------------
# /api/audio-upload route
# -----------------------

@classify_bp.route("/audio-upload", methods=["POST"])
def upload_audio():
    """Receives a WAV clip, runs model inference, returns prediction."""
    if "audio" not in request.files:
        print("ERROR: No 'audio' file part in the request.")
        return jsonify({"error": "No audio file part in the request"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        print("ERROR: No selected file in the request.")
        return jsonify({"error": "No selected file"}), 400

    wav_temp_path = None

    try:
        # Access shared model + configs from the Flask app context
        model = current_app.config["model"]
        classes = current_app.config["classes"]
        device = current_app.config["device"]
        instance_path = current_app.instance_path

        # 1. Save incoming WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=instance_path) as tmp_file:
            audio_file.save(tmp_file.name)
            wav_temp_path = Path(tmp_file.name)

        print(f"DEBUG: Received WAV file from frontend: {audio_file.filename}. Saved to {wav_temp_path}")
        print(f"DEBUG: File content type: {audio_file.content_type}")

        # 2. Preprocess and run model
        wav = load_audio(wav_temp_path)
        mel = preprocess(wav).to(device)

        with torch.no_grad():
            outputs = model(mel)
            probs = F.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)

            pred_class = classes[pred_idx.item()]
            confidence = round(conf.item() * 100, 2)

            print(f"DEBUG: Predicted: {pred_class} ({confidence:.2f}% confidence)")

        # 3. Return identical JSON structure
        return jsonify({
            "message": f"Audio file {audio_file.filename} received and processed for prediction!",
            "prediction": pred_class,
            "confidence": confidence,
        }), 200

    except Exception as e:
        print(f"ERROR: Failed to process audio: {e}")
        return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500

    finally:
        if wav_temp_path and wav_temp_path.exists():
            os.remove(wav_temp_path)
            print(f"DEBUG: Cleaned up temporary WAV file: {wav_temp_path}")