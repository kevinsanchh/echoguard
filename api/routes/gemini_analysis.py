# routes/Gemini_analysis.py

"""
Gemini_analysis.py

This module defines the Gemini wrapper endpoint.

ROLE (CURRENT MESSAGE):
- Do NOT call the real Gemini API yet.
- Act as a coordination/checkpoint layer that decides when we have
  enough data to run the rubric-based Gemini prompt.

DATA FLOW:
- Model results (environmental sound classifications) are stored
  in SessionManager under "nonspeech_results".
- Transcription results (text + segments) are stored in
  SessionManager under "transcription".

IMPORTANT ARCHITECTURAL NOTE:
- Model results and transcription will NOT arrive at the same time.
  Typically:
    - Model results arrive first (per 5-second clip, almost real-time).
    - Transcription arrives later (after all clips are processed and
      Faster-Whisper finishes).

- The Gemini endpoint must therefore:
    1. Accept a recording_id.
    2. Look up the session in SessionManager.
    3. Check whether BOTH of the following are available:
        - Non-empty transcription text.
        - At least one nonspeech_result.
    4. Return a status:
        - "waiting_for_transcription_and_model_results"
        - "waiting_for_transcription"
        - "waiting_for_model_results"
        - "ready" (both available; Gemini can safely run here later)

LATER (FUTURE MESSAGE):
- When Gemini is implemented, this endpoint will:
    - Build a dynamic prompt using BOTH:
        * transcription (text + segments)
        * model results (nonspeech_results)
    - Run the Gemini model.
    - Store + return the risk/benefit scores back to the frontend.
"""

from flask import Blueprint, request, jsonify
from utils.session_manager import get_session

gemini_bp = Blueprint("gemini", __name__, url_prefix="/process")


@gemini_bp.route("/gemini", methods=["POST"])
def gemini_check_or_analyze():
    """
    Gemini wrapper coordination endpoint.

    CURRENT BEHAVIOR (no real Gemini call yet):
    - Expects: recording_id (form or JSON).
    - Looks up the session in SessionManager.
    - Checks whether BOTH model results and transcription exist.
    - Returns a status describing what is still missing, or "ready".

    FUTURE BEHAVIOR:
    - When Gemini is wired in, this is where we'll:
        - Build the rubric-based dynamic prompt.
        - Use BOTH transcription + model results.
        - Run the Gemini model.
        - Return the final analysis to the frontend.
    """

    # 1. Extract recording_id from form-data or JSON
    recording_id = request.form.get("recording_id", type=str)

    if recording_id is None and request.is_json:
        payload = request.get_json(silent=True) or {}
        recording_id = payload.get("recording_id")

    if recording_id is None:
        print("\n[Gemini] ERROR: Missing 'recording_id' in request.\n")
        return jsonify({"error": "Missing required field 'recording_id'"}), 400

    print(f"[Gemini] Received request | recording_id={recording_id}")

    # 2. Fetch session from SessionManager
    session = get_session(recording_id)

    if session is None:
        print(f"[Gemini] WARNING: No session found for recording_id={recording_id}")
        return jsonify({
            "recording_id": recording_id,
            "status": "no_session",
            "message": "No session data found for this recording_id."
        }), 404

    # 3. Inspect transcription + nonspeech_results from the session
    transcription = session.get("transcription", {})
    transcription_text = transcription.get("text")

    nonspeech_results = session.get("nonspeech_results", [])

    has_transcription = bool(transcription_text and transcription_text.strip())
    has_model_results = bool(nonspeech_results)  # At least one nonspeech result

    # 4. Decide readiness state
    if not has_transcription and not has_model_results:
        status = "waiting_for_transcription_and_model_results"
        print(
            f"[Gemini] Status for recording_id={recording_id}: "
            f"{status} (no transcription, no model results)"
        )
    elif not has_transcription and has_model_results:
        status = "waiting_for_transcription"
        print(
            f"[Gemini] Status for recording_id={recording_id}: "
            f"{status} (model results present, transcription missing)"
        )
    elif has_transcription and not has_model_results:
        status = "waiting_for_model_results"
        print(
            f"[Gemini] Status for recording_id={recording_id}: "
            f"{status} (transcription present, no model results)"
        )
    else:
        # READY: both transcription + at least one model result exist
        status = "ready"
        print(
            f"[Gemini] Status for recording_id={recording_id}: {status} "
            f"(transcription + model results available; Gemini can run here later)"
        )

    # 5. Return status + minimal debugging info
    return jsonify({
        "recording_id": recording_id,
        "status": status,
        "has_transcription": has_transcription,
        "has_model_results": has_model_results,
        "num_model_results": len(nonspeech_results),
        # NOTE: We intentionally do NOT return transcription or model details
        # here yet; this endpoint currently acts as a readiness checker.
    }), 200
