# routes/transcribe.py

from flask import Blueprint, request, jsonify
from utils.session_manager import (
    get_all_full_clips,
    mark_session_finished,
)

transcribe_bp = Blueprint("transcribe", __name__, url_prefix="/process")


@transcribe_bp.route("/transcribe", methods=["POST"])
def transcribe_full_recording():
    """
    Stub endpoint for recording transcription.

    CURRENT BEHAVIOR:
    - Expects `recording_id` in form-data.
    - Retrieves all stored FULL CLIP paths from session_manager.
    - Logs all received clip paths in order.
    - Returns placeholder transcript JSON.
    - Marks the session as finished.

    FUTURE:
    - Will load the WAV files.
    - Will run transcription across all clips (concatenated or batched).
    - Will delete full clip files after successful transcription.
    """

    # --------------------------------------------------------
    # 1. Extract required field
    # --------------------------------------------------------
    recording_id = request.form.get("recording_id", type=str)

    if recording_id is None:
        print("\n[Transcribe] ERROR: Missing 'recording_id' in request.\n")
        return jsonify({"error": "Missing required field 'recording_id'"}), 400

    # --------------------------------------------------------
    # 2. Retrieve FULL CLIPS for this recording
    # --------------------------------------------------------
    clip_paths = get_all_full_clips(recording_id)

    print(
        f"[Transcribe] Received transcription request | recording_id={recording_id}"
    )

    if not clip_paths:
        print(
            f"[Transcribe] WARNING: No FULL CLIPS found for recording {recording_id}."
        )
        # We still return a placeholder result
        mark_session_finished(recording_id)
        return jsonify({
            "recording_id": recording_id,
            "transcript": "",
            "status": "no_clips_found"
        }), 200

    print(
        f"[Transcribe] Found {len(clip_paths)} FULL CLIPS for transcription "
        f"| recording_id={recording_id}"
    )

    # Log each clip path in order
    for clip_path in clip_paths:
        print(f"[Transcribe] FULL CLIP | path={clip_path}")

    # --------------------------------------------------------
    # 3. Placeholder response (no real transcription yet)
    # --------------------------------------------------------
    placeholder_transcript = "TRANSCRIPTION_PENDING"

    # Mark session finished
    mark_session_finished(recording_id)

    print(
        f"[Transcribe] Transcription stub complete for recording_id={recording_id}"
    )

    return jsonify({
        "recording_id": recording_id,
        "transcript": placeholder_transcript,
        "status": "ok"
    }), 200
