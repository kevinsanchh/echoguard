# routes/Gemini_analysis.py

"""
Gemini_analysis.py

This module defines the Gemini wrapper endpoint.

"""

from flask import Blueprint, request, jsonify
from utils.session_manager import get_session
from utils.config import GEMINI_API_KEY
from utils.config import GEMINI_STATIC_PROMPT

import json
import google.generativeai as genai

gemini_bp = Blueprint("gemini", __name__, url_prefix="/process")

# Configure Gemini client once
genai.configure(api_key=GEMINI_API_KEY)


def _build_gemini_prompt(transcription_text: str, cnn_results_obj) -> str:
    
    
    safe_cnn_json = json.dumps(cnn_results_obj, ensure_ascii=False)

    dynamic_section = f"""
        TRANSCRIPTION:
        <transcription>
        {transcription_text}
        </transcription>

        CNN_CLASS_RESULTS:
        <cnn_results>
        {safe_cnn_json}
        </cnn_results>
        """

    final_prompt = f"{GEMINI_STATIC_PROMPT}\n\n{dynamic_section}"
    return final_prompt.strip()


def _run_gemini_analysis(transcription_text, cnn_results_obj):
    """
    Calls Gemini 2.5 Flash and returns CLEANED JSON.
    Handles markdown fences and multiple response formats.
    """

    prompt = _build_gemini_prompt(transcription_text, cnn_results_obj)

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    # Extract raw text safely
    raw_text = None

    try:
        if hasattr(response, "text") and response.text:
            raw_text = response.text.strip()
        else:
            raw_text = (
                response.candidates[0]
                .content.parts[0]
                .text.strip()
            )
    except Exception:
        raw_text = ""

    if not raw_text:
        raise ValueError("[Gemini] Empty response from Gemini model")

    # Clean markdown fences
    cleaned = raw_text.strip()

    # strip opening ```
    if cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    # strip closing ```
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    print("\n[Gemini] CLEANED RAW OUTPUT:\n", cleaned)

    # Parse JSON
    try:
        analysis = json.loads(cleaned)
    except Exception as e:
        raise ValueError(
            f"[Gemini] Failed to parse Gemini JSON output: {e}\nRaw cleaned:\n{cleaned}"
        )

    return analysis



@gemini_bp.route("/gemini", methods=["POST"])
def gemini_check_or_analyze():
    """
    This endpoint:
    • checks readiness (transcription + model results)
    • runs Gemini analysis when ready
    • STORES result in session["final_gemini_result"]
    """

    req_data = request.form or request.json or {}
    recording_id = req_data.get("recording_id")

    if not recording_id:
        return jsonify({"error": "Missing recording_id"}), 400

    session = get_session(recording_id)

    transcription = session.get("transcription", {})
    transcription_text = transcription.get("text")
    nonspeech_results = session.get("nonspeech_results", [])

    has_transcription = bool(transcription_text and transcription_text.strip())
    has_model_results = bool(nonspeech_results)

    # If not ready, return waiting status
    if not has_transcription and not has_model_results:
        return jsonify({
            "recording_id": recording_id,
            "status": "waiting_for_transcription_and_model_results"
        })

    if not has_transcription:
        return jsonify({
            "recording_id": recording_id,
            "status": "waiting_for_transcription"
        })

    if not has_model_results:
        return jsonify({
            "recording_id": recording_id,
            "status": "waiting_for_model_results"
        })

    # READY → Run Gemini
    try:
        analysis = _run_gemini_analysis(
            transcription_text=transcription_text,
            cnn_results_obj=nonspeech_results
        )

        print(f"[Gemini] Analysis complete for recording_id={recording_id} "
              f"| risk={analysis.get('risk_score')} | benefit={analysis.get('benefit_score')}")

        # NEW: Store final Gemini result in session
        session["final_gemini_result"] = analysis
        print(f"[Gemini] Stored final Gemini result for recording_id={recording_id}")


        return jsonify({
            "recording_id": recording_id,
            "status": "completed",
            "has_transcription": True,
            "has_model_results": True,
            "num_model_results": len(nonspeech_results),
            "analysis": analysis
        })

    except Exception as e:
        return jsonify({
            "recording_id": recording_id,
            "status": "gemini_error",
            "has_transcription": has_transcription,
            "has_model_results": has_model_results,
            "num_model_results": len(nonspeech_results),
            "error": str(e)
        }), 500
