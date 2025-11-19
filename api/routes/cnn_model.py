# routes/cnn_model.py

from flask import Blueprint, request, jsonify, current_app
from pathlib import Path
import tempfile
import os

import torch

from utils.audio_utils import load_audio
from utils.pipeline_router import send_to_gemini
from utils.model_loader import (
    preprocess_waveform,
    predict,
)

model_bp = Blueprint("model", __name__, url_prefix="/process")


@model_bp.route("/model", methods=["POST"])
def classify_nonspeech_clip():
    """
    Environmental sound classification endpoint.

    EXACT same behavior as before:
    - Accepts WAV file
    - Saves to temp
    - Loads via load_audio()
    - Runs model
    - Returns JSON in old format:
        {
            "recording_id": ...,
            "clip_index": ...,
            "prediction": top_class,
            "confidence": top_conf_pct,
            "detections": [...],
            "threshold": THRESHOLD,
            "is_last_clip": ...
        }
    - Triggers Gemini on last clip
    """

    # 1. Validate request
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

    missing = []
    if recording_id is None: missing.append("recording_id")
    if clip_index is None: missing.append("clip_index")
    if is_last_raw is None: missing.append("is_last_clip")

    if missing:
        print(f"\n[Model] ERROR: Missing required fields: {', '.join(missing)}\n")
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    is_last_clip = is_last_raw.lower() == "true"

    # 2. Save temp WAV file
    instance_path = current_app.instance_path
    temp_wav_path = None

    try:
        Path(instance_path).mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=instance_path) as tmp:
            audio_file.save(tmp.name)
            temp_wav_path = Path(tmp.name)

        print(
            f"[Model] Received NON-SPEECH WAV | recording_id={recording_id}, "
            f"clip_index={clip_index} | path={temp_wav_path}"
        )

        # 3. Load model + classes + device
        model = current_app.config.get("model")
        classes = current_app.config.get("classes")
        device = current_app.config.get("device")

        if model is None or classes is None or device is None:
            print("\n[Model] ERROR: model/classes/device not found in app.config.\n")
            return jsonify({"error": "Model configuration missing on server"}), 500

        # 4. Load waveform from temp file
        waveform = load_audio(temp_wav_path)

        # waveform returned by load_audio may be shape (channels, samples)
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)

        orig_sr = current_app.config.get("sample_rate", 16000)

        # 5. preprocessing pipeline
        mel_batch = preprocess_waveform(waveform, orig_sr).to(device)

        # 6. NEW inference (multi-label sigmoid)
        THRESHOLD = 0.70
        result = predict(model, classes, mel_batch, threshold=THRESHOLD)

        top_class = result["top_class"]
        top_score = result["top_score"]       # float 0–1
        detections_raw = result["detections"] # list[(class, score)]

        top_conf_pct = round(top_score * 100.0, 2)

        print(
            f"[Model] Inference complete | recording_id={recording_id}, clip_index={clip_index} | "
            f"top_class={top_class}, top_conf={top_conf_pct:.2f}%"
        )

        # Convert detections_raw into old output shape:
        #     {"class": class_name, "confidence": pct}
        detections = [
            {
                "class": cls,
                "confidence": round(score * 100.0, 2),
            }
            for (cls, score) in detections_raw
        ]

        print(
            f"[Model] Detections >= {THRESHOLD:.2f} | "
            f"recording_id={recording_id}, clip_index={clip_index}, num_detections={len(detections)}"
        )

        # 7. Trigger Gemini if last clip
        if is_last_clip:
            print(
                f"[Model] LAST CLIP for recording_id={recording_id}. "
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
                    f"[Model] WARNING: Gemini returned None | recording_id={recording_id}"
                )

        # 8. Return JSON
        return jsonify({
            "recording_id": recording_id,
            "clip_index": clip_index,
            "prediction": top_class,
            "confidence": top_conf_pct,   # top-1 prediction, in percent
            "detections": detections,     # multi-label detections at threshold
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
        # 9. Cleanup temp file
        if temp_wav_path is not None and temp_wav_path.exists():
            try:
                os.remove(temp_wav_path)
                print(f"[Model] Cleaned up temporary WAV file: {temp_wav_path}")
            except Exception as cleanup_err:
                print(
                    f"[Model] WARNING: Failed to delete temp WAV file {temp_wav_path}: "
                    f"{cleanup_err}"
                )
