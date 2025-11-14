import os
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from flask import Blueprint, jsonify, request, current_app
from utils.audio_utils import load_audio, preprocess
from utils.session_manager import add_clip_result
from utils.session_manager import get_session

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

        # Parse tracking form fields
        recording_id = request.form.get("recording_id" , type=str)
        clip_index   = request.form.get("clip_index", type=int)
        is_last_clip_raw = request.form.get("is_last_clip")

        if recording_id is None or clip_index is None or is_last_clip_raw is None:
            print("ERROR: Missing one or more required tracking fields.")
            return jsonify({
                "error": "Missing required fields: recording_id, clip_index, and is_last_clip are required."
            }), 400

        is_last_clip = is_last_clip_raw.lower() == "true"


        # Save incoming WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=instance_path) as tmp_file:
            audio_file.save(tmp_file.name)
            wav_temp_path = Path(tmp_file.name)

        print(f"DEBUG: Received WAV file from frontend: {audio_file.filename}. Saved to {wav_temp_path}")

        # -----------------------------------------
        # Preprocess + inference (unchanged)
        # -----------------------------------------
        wav = load_audio(wav_temp_path)
        mel = preprocess(wav).to(device)

  # ------------------------------------------------------------
        # TODO (Future Multi-Label Model Upgrade — NOT ACTIVE YET)
        #
        # Once the new multi-label model is trained, this route
        # will return multiple classes with their confidence scores.
        #
        # The logic below will:
        #   1. Iterate over all class probabilities.
        #   2. Keep only those above a confidence threshold.
        #   3. Return these as a list to the frontend.
        #
        # Example future logic:
        #
        #   threshold = 0.35  # 35% confidence
        #   detections = []
        #   for idx, class_name in enumerate(classes):
        #       conf = probs[0][idx].item()
        #       if conf >= threshold:
        #           detections.append({
        #               "class": class_name,
        #               "confidence": round(conf * 100, 2)
        #           })
        #
        #   if len(detections) == 0:
        #       return jsonify({
        #           "message": "Clip processed, but no confident detections.",
        #           "detections": []
        #       }), 200
        #
        # NOTE:
        #   We are NOT using this logic right now because the model
        #   currently only outputs a single best class prediction.
        #
        # ------------------------------------------------------------
        with torch.no_grad():
            outputs = model(mel)
            probs = F.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)

            pred_class = classes[pred_idx.item()]
            confidence = round(conf.item() * 100, 2)

        print(f"DEBUG: Predicted: {pred_class} ({confidence:.2f}% confidence)")

        # -----------------------------------------
        # Store clip result in memory
        # -----------------------------------------
        try:
            add_clip_result(
                recording_id=recording_id,
                clip_index=clip_index,
                prediction=pred_class,
                confidence=confidence,
                is_last_clip=is_last_clip
            )
        except Exception as e:
            print(f"ERROR: Could not store clip in memory: {e}")
            
        if is_last_clip:
            try:
                session_data = get_session(recording_id)

                print("\n================ SESSION COMPLETE ================")
                print(f"Recording ID: {recording_id}")
                print(f"Total clips received: {len(session_data.get('clips', {}))}")

                sorted_clips = sorted(session_data["clips"], key=lambda c: c["index"])
                for clip in sorted_clips:
                    print(f"Clip {clip['index']}: {clip}")

                print("===================================================\n")

            except Exception as e:
                print(f"ERROR: Could not retrieve full session: {e}")

        # -----------------------------------------
        # Return JSON (identical)
        # -----------------------------------------
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