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
    """
    Build the final prompt for Gemini.

    Important behavior:
    - If transcription_text is None  -> this function SHOULD NOT be called.
    - If transcription_text == ""    -> transcription ran, but no speech was detected.
      In that case we:
        * put a neutral placeholder in <transcription>
        * add a SYSTEM NOTE clarifying that this is NOT the transcript, just metadata.
    """

    safe_cnn_json = json.dumps(cnn_results_obj, ensure_ascii=False)

    # Decide how to represent the transcription + metadata
    if transcription_text is None:
        # This should not normally happen if readiness checks are correct.
        visible_transcription = ""
        transcription_info = (
            "SYSTEM NOTE: Transcription has not completed for this recording. "
            "If you see this message, treat it as an internal error state."
        )
    elif str(transcription_text).strip():
        # Non-empty real transcript
        visible_transcription = transcription_text
        transcription_info = (
            "SYSTEM NOTE: Transcription occurred successfully. The text inside "
            "<transcription> is the full transcript of the user's recording."
        )
    else:
        # Transcription ran, but produced an empty string (no spoken words)
        visible_transcription = "[NO_SPOKEN_WORDS_TRANSCRIBED]"
        transcription_info = (
            "SYSTEM NOTE: Transcription occurred, and there was no speech detected "
            "in this recording. This note is not the transcription of the audio clip; "
            "it only informs you that the transcription result is effectively empty."
        )

    dynamic_section = f"""
        TRANSCRIPTION:
        <transcription>
        {visible_transcription}
        </transcription>

        TRANSCRIPTION_METADATA:
        <transcription_info>
        {transcription_info}
        </transcription_info>

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
    • checks readiness (transcription + model results / finished flag)
    • handles "not enough context" (Level 1) without calling Gemini
    • runs Gemini analysis when ready (Levels 2 and 3)
    • STORES result in session["final_gemini_result"]

    New behavior:
    - Level 1: If transcription completed AND session finished AND there is
      no transcript text AND no CNN results, we return "not_enough_context"
      and DO NOT call Gemini.
    - Level 2: Minimal context is allowed; Gemini runs, but the static prompt
      instructs it to self-assess confidence and output confidence_score
      and confidence_reasoning.
    - Level 3: Normal context; Gemini runs normally.
    """

    req_data = request.form or request.json or {}
    recording_id = req_data.get("recording_id")

    if not recording_id:
        return jsonify({"error": "Missing recording_id"}), 400

    session = get_session(recording_id)
    if session is None:
        return jsonify({"error": f"No session found for recording_id={recording_id}"}), 404

    transcription = session.get("transcription", {})
    transcription_text = transcription.get("text")
    nonspeech_results = session.get("nonspeech_results", [])
    finished = session.get("finished", False)

    # Flags:
    # - transcription_completed: transcription step finished (text no longer None)
    # - has_transcription_text: there is non-empty transcript text
    # - has_model_results: at least one CNN detection
    transcription_completed = transcription_text is not None
    has_transcription_text = bool(transcription_text and str(transcription_text).strip())
    has_model_results = bool(nonspeech_results)

    # Nothing meaningful has happened yet
    if not transcription_completed and not has_model_results and not finished:
        return jsonify({
            "recording_id": recording_id,
            "status": "waiting_for_transcription_and_model_results",
            "transcription_completed": False,
            "has_transcription": False,
            "has_model_results": False,
            "finished": finished,
        })

    # Transcription not yet completed → always wait for it
    if not transcription_completed:
        return jsonify({
            "recording_id": recording_id,
            "status": "waiting_for_transcription",
            "transcription_completed": False,
            "has_transcription": False,
            "has_model_results": has_model_results,
            "finished": finished,
        })

    # At this point transcription has completed (even if text is empty string).

    # LEVEL 1: Zero-context case.
    # Transcription ran, session finished, but:
    # - no transcript text (empty)
    # - no CNN model results
    # => Do NOT call Gemini; return "not_enough_context".
    if finished and not has_transcription_text and not has_model_results:
        print(
            f"[Gemini] NOT ENOUGH CONTEXT | recording_id={recording_id} | "
            f"empty transcript AND no model results."
        )
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
            "finished": finished,
        })

    # If we have no model results AND the session is not yet marked finished,
    # there might still be more clips / classification to come → wait.
    if not has_model_results and not finished:
        return jsonify({
            "recording_id": recording_id,
            "status": "waiting_for_model_results",
            "transcription_completed": True,
            "has_transcription": has_transcription_text,
            "has_model_results": False,
            "finished": finished,
        })

    # READY → Run Gemini.
    # Even if has_model_results == False, the "finished" flag tells us that
    # classification has effectively completed with zero usable detections.
    # This covers Level 2 (minimal context) and Level 3 (normal context).

    # Prepare CNN results object, including a default entry if empty.
    if not nonspeech_results:
        cnn_results_obj = [{
            "class": "none_detected",
            "confidence": 0.0,
            "note": (
                "No environmental sounds passed validation; the audio was primarily "
                "speech or silence, so the environmental sound model did not produce "
                "any detections."
            ),
        }]
    else:
        cnn_results_obj = nonspeech_results

    try:
        analysis = _run_gemini_analysis(
            transcription_text=transcription_text,
            cnn_results_obj=cnn_results_obj,
        )

        # Extract confidence-related fields safely
        confidence_score = analysis.get("confidence_score")
        confidence_reasoning = analysis.get("confidence_reasoning")

        low_confidence = None
        if isinstance(confidence_score, (int, float)):
            try:
                score_float = float(confidence_score)
                low_confidence = score_float < 0.4
            except (TypeError, ValueError):
                low_confidence = None

        print(
            f"[Gemini] Analysis complete for recording_id={recording_id} | "
            f"risk={analysis.get('risk_score')} | "
            f"benefit={analysis.get('benefit_score')} | "
            f"confidence_score={confidence_score} | "
            f"low_confidence={low_confidence}"
        )

        # Store final Gemini result in session
        session["final_gemini_result"] = analysis
        print(f"[Gemini] Stored final Gemini result for recording_id={recording_id}")

        return jsonify({
            "recording_id": recording_id,
            "status": "completed",
            "transcription_completed": True,
            "has_transcription": has_transcription_text,
            "has_model_results": bool(nonspeech_results),
            "num_model_results": len(nonspeech_results),
            "finished": finished,
            "analysis": analysis,
            "confidence_score": confidence_score,
            "confidence_reasoning": confidence_reasoning,
            "low_confidence": low_confidence,
        })

    except Exception as e:
        print(
            f"[Gemini] ERROR during analysis for recording_id={recording_id} | error={e}"
        )
        return jsonify({
            "recording_id": recording_id,
            "status": "gemini_error",
            "transcription_completed": transcription_completed,
            "has_transcription": has_transcription_text,
            "has_model_results": has_model_results,
            "num_model_results": len(nonspeech_results),
            "finished": finished,
            "error": str(e),
        }), 500
