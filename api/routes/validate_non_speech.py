# routes/validate_non_speech.py

from flask import Blueprint, request, jsonify, current_app
from pathlib import Path
import tempfile
import os
import torch

from utils.audio_utils import load_audio
from utils.validation_utils import validate_nonspeech_waveform

validate_bp = Blueprint("validate_non_speech", __name__, url_prefix="/process")


@validate_bp.route("/validate-non-speech", methods=["POST"])
def validate_non_speech():
    """
    Validation endpoint for stitched NON-SPEECH audio.

    ROLE:
    - Receives a NON-SPEECH waveform from VAD → router.
    - Saves it temporarily.
    - Loads it using shared audio loader.
    - Computes waveform stats (duration, RMS, max_abs, etc.).
    - Uses utils/validation_utils.py to decide validity.
    - Returns JSON {"valid": true/false}.

    IMPORTANT:
    - Invalid clips are IGNORED by the router (no model call).
    - Valid clips proceed to /process/model.
    - No predictions are generated here.
    """

    # 1. Basic presence checks
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

    # 2. Save the incoming NON-SPEECH waveform to a temp WAV
    instance_path = current_app.instance_path
    Path(instance_path).mkdir(parents=True, exist_ok=True)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".wav", dir=instance_path
        ) as tmp:
            audio_file.save(tmp.name)
            temp_path = Path(tmp.name)

        print(
            "\n" + "=" * 70 +
            f"\n[Validate] New NON-SPEECH clip received for validation | "
            f"recording_id={recording_id}, clip_index={clip_index}"
            f"\n[Validate] Saved NON-SPEECH WAV file '{audio_file.filename}' to {temp_path}"
        )

        # 3. Load waveform tensor
        waveform = load_audio(temp_path)

        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)

        if waveform.ndim == 1:  # ensure (channels, samples)
            waveform = waveform.unsqueeze(0)

        # 4. Validate waveform 
        sample_rate = current_app.config.get("sample_rate", 16000)

        is_valid, failure_reasons, stats = validate_nonspeech_waveform(
            waveform, sample_rate
        )

        print(
            f"[Validate] Waveform stats | recording_id={recording_id}, clip_index={clip_index} | "
            f"channels={stats['num_channels']}, samples={stats['num_samples']}, "
            f"duration={stats['duration_sec']:.3f}s, "
            f"max_abs={stats['max_abs']:.6f}, rms={stats['rms']:.6f}"
        )

        # 5. Log validation result
        if is_valid:
            print(
                f"[Validate] Validation result | recording_id={recording_id}, "
                f"clip_index={clip_index} | valid=True"
            )
        else:
            print(
                f"[Validate] Validation result | recording_id={recording_id}, "
                f"clip_index={clip_index} | valid=False | reasons={failure_reasons}"
            )

        print("=" * 70 + "\n")

        # 6. Return validation decision
        return jsonify({"valid": is_valid}), 200

    except Exception as e:
        print(
            f"\n[Validate] ERROR: Failed to validate NON-SPEECH clip | "
            f"recording_id={recording_id}, clip_index={clip_index} | error={e}\n"
        )
        return jsonify({"error": f"Validation failed: {str(e)}"}), 500

    finally:
        # Always clean up temp file
        if temp_path is not None and temp_path.exists():
            try:
                os.remove(temp_path)
                print(f"[Validate] Cleaned up temp file: {temp_path}")
            except Exception as cleanup_err:
                print(f"[Validate] WARNING: Failed to delete temp file {temp_path}: {cleanup_err}")
