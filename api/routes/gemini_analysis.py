# routes/Gemini_analysis.py

"""
Gemini_analysis.py

This module defines the Gemini wrapper endpoint.

CURRENT ROLE:
- Acts as a coordination layer that:
    * Checks if transcription + model results are ready.
    * When both are present, runs the Gemini 2.5-Flash model
      with a rubric-based prompt to compute risk/benefit scores.
    * Returns a structured JSON analysis to the frontend.
    * Clears session data after a successful analysis.

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
    4. If either is missing, return a status:
        - "waiting_for_transcription_and_model_results"
        - "waiting_for_transcription"
        - "waiting_for_model_results"
       If both are present, run Gemini and return status="completed".
"""

from flask import Blueprint, request, jsonify
from utils.session_manager import get_session
from utils.config import GEMINI_API_KEY
from utils.pipeline_router import send_gemini_result_to_frontend

import json
import google.generativeai as genai

gemini_bp = Blueprint("gemini", __name__, url_prefix="/process")

# Configure Gemini client once
genai.configure(api_key=GEMINI_API_KEY)


def _build_gemini_prompt(transcription_text: str, cnn_results_obj) -> str:
    """
    Build the full Gemini prompt using a safe structured format.
    100% of the user's original prompt is preserved.
    """
    
    # Safe JSON encoding for CNN results (no indentation, no escaping issues)
    safe_cnn_json = json.dumps(cnn_results_obj, ensure_ascii=False)

    prompt = f"""
You are the reasoning and interpretation module for EchoGuard, an AI system built to help parents monitor the safety and content of what their children are exposed to by analyzing audio from media (such as TV shows, online videos, or games) and determining whether the environment or content presents potentially harmful, violent, or disturbing material.

EchoGuard’s backend processes audio through multiple stages before information reaches you. You never receive raw audio. Instead, you always receive two processed inputs:

1. A full transcription of all spoken content in the recording.
2. A set of model-generated environmental sound classifications for small audio segments (3–5 seconds each). These detections come from EchoGuard’s convolutional neural network (CNN) classifier.

The CNN model is trained on the following unique sound classes:

Multi-label training exists, but is not directly used here. The following are the unique classes you must consider:

- Crying
- Explosion
- Glass Shattering
- Fire
- Emergency Siren
- Artillery Fire
- Battle Cry
- Car Alarm
- Gunfire
- Slap/Smack
- Chainsaw
- Thunderstorm
- Growling
- Roar
- Whimper
- Screaming

In your analysis, treat the transcript as the primary and most reliable source of information. Spoken content provides the clearest picture of intent, emotion, and context. When the transcript is present and reasonably detailed, it should guide the majority of your understanding.

The CNN sound detections should be treated as supporting context, not absolute truth. These predictions are helpful but not perfectly accurate. Their purpose is to help you refine, confirm, or deepen your interpretation of the transcript — not to override it. Use them to validate or adjust your understanding when appropriate.

### Dynamic Inputs
Below are the dynamic inputs you must use for your analysis:

TRANSCRIPTION:
<transcription>
{transcription_text}
</transcription>

CNN_CLASS_RESULTS (JSON):
<cnn_results>
{safe_cnn_json}
</cnn_results>

### Risk and Benefit Scoring Rubric

Your primary job is to generate two numerical scores (risk and benefit) that accurately represent the content of the audio, with risk being the higher-priority metric. These scores must reflect the purpose of EchoGuard: helping parents understand whether the media their children are exposed to contains violent, disturbing, unsafe, or otherwise harmful content.

#### Purpose-Aligned Risk Interpretation
Before applying any sound-based adjustments, you must first determine a baseline risk score using your own reasoning. This baseline score should come from analyzing the transcript (primary signal) and the contextual meaning of the sound detections (secondary signal). The baseline should be realistic and aligned with EchoGuard’s purpose.

This means:
- High-violence content should never result in a low risk score.
- If the transcript or the overall scenario strongly suggests danger, weapons, abuse, threats, injury, or emotional distress, the baseline risk should be correspondingly high.
- Conversely, neutral or positive content should have a low baseline score.
- The transcript should always guide the majority of the interpretation.

After you determine the baseline, then you will refine it using the class-based scoring rules below.

#### Class-Based Risk Add-Ons (Supporting, Not Dominant)
EchoGuard’s CNN classifier may detect specific environmental sounds. These detections should refine your baseline risk score, not replace it. Because the CNN model is not perfectly accurate, detections should be treated as helpful signals rather than absolute truths.

Use the following unique sound classes, grouped into three categories:

Category 1 (Most Violent — +10% base):
- Gunfire
- Artillery Fire
- Explosion
- Chainsaw
- Slap/Smack

Category 2 (Moderately Violent — +6% base):
- Glass Shattering
- Fire
- Battle Cry
- Car Alarm
- Emergency Siren
- Roar

Category 3 (Low-Level Distress — +3% base):
- Crying
- Whimpering
- Growling
- Thunderstorm
- Screaming (non-chainsaw)

When adding class-based risk adjustments, use this formula:

AdditionalRisk = BaseValue * ConfidenceScore * 0.9

Add the total additional risk to the baseline risk score.
Cap the final result at 100.

#### Benefit Score
The benefit score is secondary and does not use class-based adjustments.
Use your reasoning to determine whether the content contains positive, educational, friendly, humorous, or otherwise beneficial themes.
If the media contains both harmful and beneficial aspects, the scores may reflect both accordingly.

#### Final Output Requirements
After completing all analysis and adjustments, output:

- risk_score (0–100)
- benefit_score (0–100)
- risk_reasoning paragraph (1–3 sentences)
- benefit_reasoning paragraph (1–3 sentences)
- steps field summarizing your overall reasoning process

You must output valid JSON only. Do not include any text outside a JSON object.
"""
    return prompt.strip()


def _run_gemini_analysis(transcription_text: str, cnn_results_obj):
    """
    Call Gemini 2.5-Flash, extract the output safely, remove Markdown fences,
    and parse the JSON. Fully compatible with google-generativeai 0.8.5.
    """

    # Encode CNN results cleanly
    safe_cnn_json = json.dumps(cnn_results_obj, ensure_ascii=False)

    # Build the full prompt
    prompt = _build_gemini_prompt(
        transcription_text=transcription_text,
        cnn_results_obj=safe_cnn_json
    )

    model = genai.GenerativeModel("gemini-2.5-flash")

    # ---- Make request ----
    response = model.generate_content(prompt)

    # ---- Extract text safely (SDK 0.8.5 behavior) ----
    raw_text = None

    # Preferred source when available
    if hasattr(response, "text") and response.text:
        raw_text = response.text.strip()

    # Fallback for Gemini responses that use candidates/parts
    if not raw_text:
        try:
            raw_text = (
                response.candidates[0]
                .content.parts[0]
                .text.strip()
            )
        except Exception:
            raw_text = ""

    # ---- Handle no output at all ----
    if not raw_text:
        raise ValueError("[Gemini] Empty response from Gemini model")

    cleaned = raw_text.strip()

    # Remove opening fences
    if cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()  # strip leading triple backticks

        # Optional "json" language tag
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    # Remove closing fences
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    # Debug: show cleaned output
    print("\n[Gemini] CLEANED RAW OUTPUT:\n", cleaned)

    # ---- Parse cleaned JSON ----
    try:
        analysis = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"[Gemini] Failed to parse Gemini JSON output: {e}\n\n"
            f"RAW OUTPUT WAS:\n{raw_text}\n\n"
            f"CLEANED OUTPUT WAS:\n{cleaned}"
        )

    # ---- Validate required fields ----
    required_keys = [
        "risk_score",
        "benefit_score",
        "risk_reasoning",
        "benefit_reasoning",
        "steps",
    ]
    for key in required_keys:
        if key not in analysis:
            raise ValueError(f"[Gemini] Missing required key in analysis: {key}")

    return analysis


@gemini_bp.route("/gemini", methods=["POST"])
def gemini_check_or_analyze():
    """
    Gemini wrapper endpoint.

    BEHAVIOR:
    - Expects: recording_id (form or JSON).
    - Looks up the session in SessionManager.
    - Checks whether BOTH model results and transcription exist.
    - If either is missing, returns a waiting status exactly as before.
    - If both are present:
        * Builds a rubric-based prompt.
        * Calls Gemini 2.5-Flash.
        * Parses the JSON-only output.
        * Returns a structured analysis and marks status="completed".
        * Clears session data for this recording_id.
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

    # If either piece is missing, keep the old readiness behavior exactly
    if not has_transcription and not has_model_results:
        status = "waiting_for_transcription_and_model_results"
        print(
            f"[Gemini] Status for recording_id={recording_id}: "
            f"{status} (no transcription, no model results)"
        )
        return jsonify({
            "recording_id": recording_id,
            "status": status,
            "has_transcription": has_transcription,
            "has_model_results": has_model_results,
            "num_model_results": len(nonspeech_results),
        }), 200

    if not has_transcription and has_model_results:
        status = "waiting_for_transcription"
        print(
            f"[Gemini] Status for recording_id={recording_id}: "
            f"{status} (model results present, transcription missing)"
        )
        return jsonify({
            "recording_id": recording_id,
            "status": status,
            "has_transcription": has_transcription,
            "has_model_results": has_model_results,
            "num_model_results": len(nonspeech_results),
        }), 200

    if has_transcription and not has_model_results:
        status = "waiting_for_model_results"
        print(
            f"[Gemini] Status for recording_id={recording_id}: "
            f"{status} (transcription present, no model results)"
        )
        return jsonify({
            "recording_id": recording_id,
            "status": status,
            "has_transcription": has_transcription,
            "has_model_results": has_model_results,
            "num_model_results": len(nonspeech_results),
        }), 200

    # At this point, we have BOTH transcription + model results
    print(
        f"[Gemini] Status for recording_id={recording_id}: ready "
        f"(transcription + model results available; running Gemini analysis)"
    )

    try:
        analysis = _run_gemini_analysis(
            transcription_text=transcription_text,
            cnn_results_obj=nonspeech_results,
        )
        print(
            f"[Gemini] Analysis complete for recording_id={recording_id} | "
            f"risk_score={analysis.get('risk_score')}, "
            f"benefit_score={analysis.get('benefit_score')}"
        )

        # Clear session data after successful analysis
        session.clear()
        print(f"[Gemini] Cleared session data for recording_id={recording_id}")

        # --- NEW LOGIC: shape Gemini result like CNN ---
        frontend_payload = send_gemini_result_to_frontend(recording_id, analysis)

        # Return clean final payload to frontend
        return jsonify(frontend_payload), 200

    except Exception as e:
        print(
            f"[Gemini] ERROR: Failed to run Gemini analysis for recording_id={recording_id} | "
            f"error={e}"
        )
        return jsonify({
            "recording_id": recording_id,
            "status": "gemini_error",
            "has_transcription": has_transcription,
            "has_model_results": has_model_results,
            "num_model_results": len(nonspeech_results),
            "error": str(e),
        }), 500
