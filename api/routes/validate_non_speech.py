# routes/validate_non_speech.py

from flask import Blueprint, request, jsonify

validate_bp = Blueprint("validate_non_speech", __name__, url_prefix="/process")


@validate_bp.route("/validate-non-speech", methods=["POST"])
def validate_non_speech():
    """
     Validation endpoint for stitched NON-SPEECH audio.

    CURRENT BEHAVIOR (stub / scaffold):
    - Ensures the request contains:
        - files["audio"]
        - form["recording_id"]
        - form["clip_index"]
    - Logs these values.
    - Always returns {"valid": true} for now.

    FUTURE:
    - We can inspect the waveform here (e.g., energy, duration)
      and reject silence / low-information clips.
    """

    # --------------------------------------------------------
    # 1. Basic presence checks
    # --------------------------------------------------------
    if "audio" not in request.files:
        print("\n[Validate] ERROR: No 'audio' file part in the request.\n")
        return jsonify({"error": "No audio file part in the request"}), 400

    audio_file = request.files["audio"]
    recording_id = request.form.get("recording_id", type=str)
    clip_index = request.form.get("clip_index", type=int)

    missing_fields = []
    if recording_id is None:
        missing_fields.append("recording_id")
    if clip_index is None:
        missing_fields.append("clip_index")

    if missing_fields:
        print(
            f"\n[Validate] ERROR: Missing required fields: {', '.join(missing_fields)}\n"
        )
        return jsonify({
            "error": f"Missing required fields: {', '.join(missing_fields)}"
        }), 400

    # --------------------------------------------------------
    # 2. Logging for now (no real audio inspection yet)
    # --------------------------------------------------------
    print(
        f"[Validate] Received NON-SPEECH clips for validation | "
        f"recording_id={recording_id}, clip_index={clip_index}, "
        f"filename='{audio_file.filename}'"
    )

    # NOTE: In the future, we can load+analyze the audio to decide validity.
    # For now, always mark as valid so the model endpoint is always hit.
    is_valid = True

    print(
        f"[Validate] Validation result for recording_id={recording_id}, "
        f"clip_index={clip_index} | valid={is_valid}"
    )

    return jsonify({
        "valid": is_valid
    }), 200
