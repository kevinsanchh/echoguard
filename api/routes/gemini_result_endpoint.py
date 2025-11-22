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

    # Detect "Not Enough Context" Scenario

    transcription = session.get("transcription", {})
    transcription_text = transcription.get("text")
    nonspeech_results = session.get("nonspeech_results", [])
    finished = session.get("finished", False)

    transcription_completed = transcription_text is not None
    has_transcription_text = bool(transcription_text and str(transcription_text).strip())
    has_model_results = bool(nonspeech_results)

    # Gemini NEVER ran because there was not enough context.
    if (
        transcription_completed
        and finished
        and not has_transcription_text
        and not has_model_results
    ):
        return jsonify({
            "recording_id": recording_id,
            "status": "not_enough_context",
            "message": (
                "No meaningful speech or environmental audio was detected in this "
                "recording. Please try recording again with more content."
            ),
            "transcription_completed": True,
            "has_transcription": False,
            "has_model_results": False,
            "finished": True,
        }), 200

    # NORMAL PATH: Check for stored Gemini result

    gemini_result = session.get("final_gemini_result")

    # If no Gemini result exists, but we are not in Level-1, it means Gemini:
    #   • either hasn't run yet
    #   • or is still waiting for transcript or model results
    if not gemini_result:
        return jsonify({
            "recording_id": recording_id,
            "status": "not_ready",
            "message": "Gemini result not yet available.",
            "transcription_completed": transcription_completed,
            "has_transcription": has_transcription_text,
            "has_model_results": has_model_results,
            "finished": finished,
        }), 200


    # READY — Prepare structured result for frontend
    response_payload = {
        "recording_id": recording_id,
        "status": "ready",
        "gemini_result": {
            "risk_score": gemini_result.get("risk_score"),
            "benefit_score": gemini_result.get("benefit_score"),
            "risk_reasoning": gemini_result.get("risk_reasoning"),
            "benefit_reasoning": gemini_result.get("benefit_reasoning"),
            "confidence_score": gemini_result.get("confidence_score"),
            "confidence_reasoning": gemini_result.get("confidence_reasoning"),
        },
    }

    # Remove the Gemini result so it is not returned twice
    try:
        del session["final_gemini_result"]
        print(f"[Gemini Result] Cleared final Gemini result for recording_id={recording_id}")
    except Exception:
        print(f"[Gemini Result] WARNING: Could not delete Gemini result for recording_id={recording_id}")

    return jsonify(response_payload), 200
