# routes/gemini_result_endpoint.py

from flask import Blueprint, request, jsonify
from utils.session_manager import get_session

gemini_result_bp = Blueprint("gemini_result", __name__, url_prefix="/process")


@gemini_result_bp.route("/gemini_result", methods=["GET"])
def get_gemini_result():

    recording_id = request.args.get("recording_id")

    if not recording_id:
        return jsonify({"error": "Missing required parameter 'recording_id'"}), 400

    session = get_session(recording_id)

    if session is None:
        return jsonify({
            "recording_id": recording_id,
            "status": "no_session",
            "message": "No session data found for this recording_id."
        }), 404

    gemini_result = session.get("final_gemini_result")

    if not gemini_result:
        return jsonify({
            "recording_id": recording_id,
            "status": "not_ready",
            "message": "Gemini result not yet available."
        }), 200

    
    # Prepare response payload (clean structure for frontend)
    response_payload = {
        "recording_id": recording_id,
        "status": "ready",
        "gemini_result": {
            "risk_score": gemini_result.get("risk_score"),
            "benefit_score": gemini_result.get("benefit_score"),
            "risk_reasoning": gemini_result.get("risk_reasoning"),
            "benefit_reasoning": gemini_result.get("benefit_reasoning"),
        },
    }

    # ------------------------------------------------------------
    # Remove the Gemini result so it is not returned twice
    # (session keeps speech, model results, etc.)
    # ------------------------------------------------------------
    try:
        del session["final_gemini_result"]
        print(f"[Gemini Result] Cleared final Gemini result for recording_id={recording_id}")
    except Exception:
        print(f"[Gemini Result] WARNING: Could not delete Gemini result for recording_id={recording_id}")

    return jsonify(response_payload), 200