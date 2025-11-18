# routes/cnn_model.py

from flask import Blueprint, request, jsonify, current_app
from pathlib import Path
import tempfile
import os

import torch
import torch.nn.functional as F

from utils.audio_utils import load_audio
from utils.pipeline_router import send_to_gemini

# If SAMPLE_RATE is defined in a shared config, import it.
# Adjust this import if your SAMPLE_RATE lives elsewhere.
from utils.config import SAMPLE_RATE

model_bp = Blueprint("model", __name__, url_prefix="/process")


def preprocess(wav):
    """
    Convert waveform into a normalized mel spectrogram.

    This is the exact method you provided, just placed here
    so the model endpoint has everything it needs in one file.
    """
    import torchaudio

    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
    )

    mel = transform(wav)
    mel_db = torchaudio.functional.amplitude_to_DB(
        mel,
        multiplier=10.0,
        amin=1e-10,
        db_multiplier=0.0,
    )

    # Normalize to mean 0, std 1
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    # Repeat to create 3-channel input for ResNet
    mel_db = mel_db.repeat(3, 1, 1)

    # Add batch dimension (1, 3, H, W)
    return mel_db.unsqueeze(0)


@model_bp.route("/model", methods=["POST"])
def classify_nonspeech_clip():
    """
    Environmental sound classification endpoint.

    EXPECTED REQUEST (from pipeline_router.route_non_speech_for_classification):
        files = {
            "audio": ("nonspeech.wav", <file-like>, "audio/wav")
        }
        data = {
            "recording_id": <str>,
            "clip_index": <int>,
            "is_last_clip": <"true" | "false">
        }

    BEHAVIOR:
    - Validate request fields & save incoming WAV to a temp file.
    - Load waveform with load_audio().
    - Preprocess with your mel-spectrogram `preprocess(wav)` function.
    - Run model → softmax probabilities over classes.
    - Build:
        * `prediction` (top-1 class name)
        * `confidence` (top-1 % for router/SessionManager/Gemini)
        * `detections` list for all classes with prob >= 0.7
    - If `is_last_clip == true`, trigger Gemini via pipeline_router.send_to_gemini(recording_id).
    - Return JSON to the router; router is responsible for:
        * calling add_nonspeech_result(...)
        * exposing anything to the frontend via /process/vad.
    """

    # 1. Basic presence checks
    if "audio" not in request.files:
        print("\n[Model] ERROR: No 'audio' file part in the request.\n")
        return jsonify({"error": "No audio file part in the request"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        print("\n[Model] ERROR: No selected file in the request.\n")
        return jsonify({"error": "No selected file"}), 400

    recording_id = request.form.get("recording_id", type=str)
    clip_index = request.form.get("clip_index", type=int)
    is_last_raw = request.form.get("is_last_clip")

    missing_fields = []
    if recording_id is None:
        missing_fields.append("recording_id")
    if clip_index is None:
        missing_fields.append("clip_index")
    if is_last_raw is None:
        missing_fields.append("is_last_clip")

    if missing_fields:
        print(
            f"\n[Model] ERROR: Missing required fields: {', '.join(missing_fields)}\n"
        )
        return jsonify({
            "error": f"Missing required fields: {', '.join(missing_fields)}"
        }), 400

    is_last_clip = is_last_raw.lower() == "true"

    instance_path = current_app.instance_path
    temp_wav_path = None

    try:
        # 2. Save incoming WAV to a temp file
        Path(instance_path).mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".wav", dir=instance_path
        ) as tmp:
            audio_file.save(tmp.name)
            temp_wav_path = Path(tmp.name)

        print(
            f"[Model] Received NON-SPEECH WAV for classification | "
            f"recording_id={recording_id}, clip_index={clip_index} | path={temp_wav_path}"
        )

        # 3. Load shared model + config
        model = current_app.config.get("model")
        classes = current_app.config.get("classes")
        device = current_app.config.get("device")

        if model is None or classes is None or device is None:
            print(
                "\n[Model] ERROR: model/classes/device missing in app.config.\n"
            )
            return jsonify({"error": "Model configuration missing on server"}), 500

        # 4. Load waveform and preprocess
        wav = load_audio(temp_wav_path)
        mel = preprocess(wav).to(device)

        # 5. Model inference → softmax probabilities
        model.eval()
        with torch.no_grad():
            logits = model(mel)
            probs = F.softmax(logits, dim=1)  # shape: [1, num_classes]

        probs_1d = probs[0]  # tensor [num_classes]

        # 6. Top-1 prediction (for SessionManager/Gemini via router)
        top_idx = int(torch.argmax(probs_1d).item())
        top_class = classes[top_idx]
        top_prob = float(probs_1d[top_idx].item())
        top_conf_pct = round(top_prob * 100.0, 2)

        print(
            f"[Model] Inference complete | recording_id={recording_id}, clip_index={clip_index} | "
            f"top_class={top_class}, top_conf={top_conf_pct:.2f}%"
        )

        # 7. Build detections list: all classes with prob >= 0.7
        THRESHOLD = 0.7
        detections = []

        for idx, class_name in enumerate(classes):
            p = float(probs_1d[idx].item())
            if p >= THRESHOLD:
                detections.append({
                    "class": class_name,
                    "confidence": round(p * 100.0, 2),  # in percent
                })

        print(
            f"[Model] Detections >= {THRESHOLD:.2f} | "
            f"recording_id={recording_id}, clip_index={clip_index}, "
            f"num_detections={len(detections)}"
        )

        # 8. If this was the last clip, trigger Gemini via the router
        if is_last_clip:
            print(
                f"[Model] LAST CLIP detected for recording_id={recording_id}. "
                f"Triggering Gemini wrapper..."
            )
            gemini_result = send_to_gemini(recording_id)
            if gemini_result is not None:
                print(
                    f"[Model] Gemini wrapper response | recording_id={recording_id} | "
                    f"status={gemini_result.get('status')}"
                )
            else:
                print(
                    f"[Model] WARNING: Gemini trigger returned None | "
                    f"recording_id={recording_id}"
                )

        # 9. Return JSON to router
        #    NOTE: router.route_non_speech_for_classification() expects:
        #       - prediction (top-1)
        #       - confidence (top-1, in percent)
        #    and will call add_nonspeech_result(...) itself.
        return jsonify({
            "recording_id": recording_id,
            "clip_index": clip_index,
            "prediction": top_class,
            "confidence": top_conf_pct,   # in percent, for SessionManager/Gemini
            "detections": detections,     # for future FE use via /process/vad
            "threshold": THRESHOLD,
            "is_last_clip": is_last_clip,
        }), 200

    except Exception as e:
        print(
            f"\n[Model] ERROR: Failed to classify NON-SPEECH clip | "
            f"recording_id={recording_id}, clip_index={clip_index} | error={e}\n"
        )
        return jsonify({
            "error": f"Failed to classify NON-SPEECH audio: {str(e)}"
        }), 500

    finally:
        if temp_wav_path is not None and temp_wav_path.exists():
            try:
                os.remove(temp_wav_path)
                print(f"[Model] Cleaned up temporary WAV file: {temp_wav_path}")
            except Exception as cleanup_err:
                print(
                    f"[Model] WARNING: Failed to delete temp WAV file {temp_wav_path}: "
                    f"{cleanup_err}"
                )
