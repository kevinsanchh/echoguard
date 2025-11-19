# routes/gemini_analysis.py

from flask import Blueprint, request, jsonify
from utils.session_manager import get_session
from utils.session_manager import session_recordings  # for cleanup
from utils.config import GEMINI_API_KEY, GEMINI_PROMPT
import google.generativeai as genai
import json
import traceback

gemini_bp = Blueprint("gemini", __name__, url_prefix="/process")

# Configure Gemini client once
genai.configure(api_key=GEMINI_API_KEY)

@gemini_bp.route("/gemini", methods=["POST"])
def gemini_check_or_analyze():
    """
    Gemini wrapper endpoint with readiness logic + real Gemini analysis.
    """
    recording_id = request.form.get("recording_id", type=str)

    if recording_id is None and request.is_json:
        payload = request.get_json(silent=True) or {}
        recording_id = payload.get("recording_id")

    if not recording_id:
        return jsonify({"error": "Missing required field 'recording_id'"}), 400

    print(f"[Gemini] Received request | recording_id={recording_id}")

    # Retrieve session
    session = get_session(recording_id)
    if session is None:
        print(f"[Gemini] No session found for {recording_id}")
        return jsonify({
            "recording_id": recording_id,
            "status": "no_session",
            "message": "No session data found for this recording_id."
        }), 404

    # Check readiness
    transcription = session.get("transcription", {})
    transcription_text = transcription.get("text")
    nonspeech_results = session.get("nonspeech_results", [])

    has_transcription = bool(transcription_text and transcription_text.strip())
    has_model_results = bool(nonspeech_results)

    if not has_transcription and not has_model_results:
        status = "waiting_for_transcription_and_model_results"
        return jsonify({
            "recording_id": recording_id,
            "status": status,
            "has_transcription": has_transcription,
            "has_model_results": has_model_results,
            "num_model_results": 0
        }), 200

    if not has_transcription:
        status = "waiting_for_transcription"
        return jsonify({
            "recording_id": recording_id,
            "status": status,
            "has_transcription": has_transcription,
            "has_model_results": has_model_results,
            "num_model_results": len(nonspeech_results)
        }), 200

    if not has_model_results:
        status = "waiting_for_model_results"
        return jsonify({
            "recording_id": recording_id,
            "status": status,
            "has_transcription": has_transcription,
            "has_model_results": has_model_results,
            "num_model_results": 0
        }), 200

    # Both available → run Gemini
    print(f"[Gemini] READY → Running Gemini analysis for recording_id={recording_id}")

    # Build prompt from template
    prompt_template = GEMINI_PROMPT

    prompt = (
        prompt_template
        .replace("[[TRANSCRIPTION]]", transcription_text)
        .replace("[[CNN_RESULTS]]", json.dumps(nonspeech_results, indent=2))
    )

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        # Extract Gemini output (text)
        result_text = response.text.strip()

        # Must be valid JSON
        analysis_json = json.loads(result_text)

    except Exception as e:
        traceback.print_exc()
        print(f"[Gemini] ERROR during Gemini analysis: {e}")

        return jsonify({
            "recording_id": recording_id,
            "status": "gemini_error",
            "error": str(e),
            "raw_output": response.text if 'response' in locals() else None
        }), 500

    # CLEANUP session after successful analysis
    if recording_id in session_recordings:
        del session_recordings[recording_id]
        print(f"[Gemini] Session {recording_id} fully cleaned after analysis.")

    # Final structured response
    return jsonify({
        "recording_id": recording_id,
        "status": "completed",
        "has_transcription": True,
        "has_model_results": True,
        "num_model_results": len(nonspeech_results),
        "analysis": analysis_json
    }), 200
