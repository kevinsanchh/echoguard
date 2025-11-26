"""
pipeline_router.py

This module orchestrates all routing actions between:
- VAD output
- validation endpoint
- model endpoint
- transcription endpoint
- session manager

NO actual classification or transcription happens here.
This file only sends data to the correct endpoints and stores intermediate state.
"""

import requests
from pathlib import Path
import torch
import requests
from utils.session_manager import (
    add_speech_segments,
    add_nonspeech_result,
    mark_session_finished,
    get_all_full_clips,
)

# -----------------------------------------------------------
# CONFIG (these will eventually be loaded from Flask config)
# ----------------------------------------------------------

VALIDATION_URL = "http://localhost:8080/process/validate-non-speech"
MODEL_URL = "http://localhost:8080/process/model"
TRANSCRIBE_URL = "http://localhost:8080/process/transcribe"


# -----------------------------------------------------------
# STORE NON-SPEECH (stitched waveform)
# -----------------------------------------------------------

def store_nonspeech_segment(recording_id, clip_index):
       """
    We DO NOT store non-speech waveforms in memory.
    The only thing that will ever be stored for a non-speech clip
    is the MODEL RESULT after classification.

    This function exists purely for architectural clarity:
    - VAD extracts the stitched non-speech waveform
    - VAD then informs the router that this clip has non-speech ready
    - The router then immediately routes the waveform through:
        validate → model → store result

    No audio storage occurs here.
    """
# print(
#         f"[Router] NON-SPEECH segment detected | "
#         f"recording={recording_id}, clip={clip_index} | ready for validation/classification"
#     )


# STORE SPEECH SEGMENTS (raw speech)

def store_speech_segment(recording_id, clip_index, segments):
    """
    Store raw speech segments extracted from VAD.
    """
    add_speech_segments(recording_id, clip_index, segments)
    print(f"[Router] Stored SPEECH segments for recording={recording_id}, clip={clip_index}")


# ROUTE NON-SPEECH → VALIDATION → MODEL → STORE RESULT

def route_non_speech_for_classification(recording_id, clip_index, waveform, is_last_clip=False):
    """
    Full pipeline for NON-SPEECH classification:
    1. Convert stitched NON-SPEECH tensor into WAV bytes.
    2. Send to validation endpoint.
    3. If valid → send to model endpoint.
    4. If invalid → skip (do NOT store or send placeholder results).
    5. Return model result or None.

    NOTE:
    - REAL inference happens in the model endpoint.
    - Waveform is a PyTorch tensor (channels, samples).
    """

    print(f"[Router] Routing NON-SPEECH for classification | recording={recording_id}, clip={clip_index}")

    # --------------------------------------------------------------
    # Step 1: Convert tensor → WAV bytes (in-memory)
    # --------------------------------------------------------------
    import io
    import torchaudio

    try:
        buffer = io.BytesIO()
        sample_rate = 16000  # consistent with VAD & loading

        # Ensure correct shape
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        torchaudio.save(buffer, waveform, sample_rate, format="wav")
        buffer.seek(0)

    except Exception as e:
        print(f"[Router] ERROR encoding NON-SPEECH to WAV bytes: {e}")
        return None

    # Step 2: Validate non-speech
    try:
        response = requests.post(
            VALIDATION_URL,
            files={"audio": ("nonspeech.wav", buffer, "audio/wav")},
            data={
                "recording_id": recording_id,
                "clip_index": clip_index
            }
        )
        response_data = response.json()
    except Exception as e:
        print(f"[Router] ERROR contacting validation endpoint: {e}")
        return None

    # Step 3: Handle validation result
    if not response_data.get("valid", False):
        print(
            f"[Router] Validation FAILED for NON-SPEECH | "
            f"recording={recording_id}, clip={clip_index}. "
            f"Skipping model classification."
        )
        # IMPORTANT: Per architecture, DO NOT send placeholder results.
        return None

    print(
        f"[Router] Validation PASSED for NON-SPEECH | "
        f"recording={recording_id}, clip={clip_index}. Sending to model endpoint."
    )

    # Step 4: Send to model endpoint
    try:
        buffer.seek(0)  # rewind for model
        model_response = requests.post(
            MODEL_URL,
            files={"audio": ("nonspeech.wav", buffer, "audio/wav")},
            data={
                "recording_id": recording_id,
                "clip_index": clip_index,
                "is_last_clip": str(is_last_clip),
            }
        )
        model_data = model_response.json()

    except Exception as e:
        print(f"[Router] ERROR contacting model endpoint: {e}")
        return None

    # Step 5: Store model result
    prediction = model_data.get("prediction")
    confidence = model_data.get("confidence")

    add_nonspeech_result(
        recording_id,
        clip_index,
        prediction,
        confidence,
        is_last_clip=is_last_clip
    )

    return model_data


# ROUTE SPEECH → TRANSCRIPTION
def send_full_clips_to_transcription(recording_id):
    """
    NEW FINAL METHOD:
    Route ALL stored full 5-second RAW WAV clips to the transcription endpoint.

    IMPORTANT:
    - We do NOT combine, decode, or process these clips here.
    - We only gather paths and send them.
    - Real transcription logic will be implemented later.
    """

    # Logging — identical format to old function
    print(f"[Router] Routing FULL CLIPS for transcription | recording={recording_id}")

    # Retrieve all full clip file paths in sorted order
    clip_paths = get_all_full_clips(recording_id)

    print(f"[Router] Found {len(clip_paths)} FULL CLIPS for transcription | recording={recording_id}")

    for idx, path in enumerate(clip_paths):
        print(f"[Router] FULL CLIP {idx} | path={path}")

    try:
        response = requests.post(
            TRANSCRIBE_URL,
            data={"recording_id": recording_id}
        )
        result = response.json()
    except Exception as e:
        print(f"[Router] ERROR contacting transcription endpoint: {e}")
        return None
    # Mark session finished for now
    mark_session_finished(recording_id)

    print(f"[Router] Transcription trigger completed for recording={recording_id} (FULL CLIPS).")

    # No response yet — transcription not implemented
    return result

def send_to_gemini(recording_id):
    """
    Trigger the Gemini wrapper after transcription finishes.

    Behavior:
    - Sends only 'recording_id'.
    - Does NOT assume transcription is ready.
    - Returns Gemini readiness status:
         * waiting_for_transcription_and_model_results
         * waiting_for_transcription
         * waiting_for_model_results
         * ready
    - No Gemini logic is executed here (wrapper only).
    """

    print(f"[Router] Routing to GEMINI endpoint | recording={recording_id}")

    try:
        response = requests.post(
            "http://localhost:8080/process/gemini",
            data={"recording_id": recording_id}
        )
        result = response.json()

        print(
            f"[Router] Gemini response for recording={recording_id}: "
            f"status={result.get('status')}, "
            f"has_transcription={result.get('has_transcription')}, "
            f"has_model_results={result.get('has_model_results')}"
        )

        return result

    except Exception as e:
        print(f"[Router] ERROR contacting Gemini endpoint: {e}")
        return None
    
def send_cnn_model_result_to_frontend(recording_id, clip_index, classification_result):
    """
    Prepare a clean JSON packet from the model result to send back to the frontend.
    This does NOT talk to the client directly; it just shapes the data.
    """
    if classification_result is None:
        return None

    return {
        "recording_id": recording_id,
        "clip_index": clip_index,
        "prediction": classification_result.get("prediction"),
        "confidence": classification_result.get("confidence"),
        "detections": classification_result.get("detections", []),
        "threshold": classification_result.get("threshold"),
        "is_last_clip": classification_result.get("is_last_clip", False),
    }

def classify_nonspeech_for_upload(waveform, sample_rate: int = 16000):
    """
    Classify NON-SPEECH audio for the upload workflow.

    Behavior:
    - Takes a stitched NON-SPEECH waveform tensor: shape (channels, samples).
    - Splits it into <= 5 second chunks based on sample_rate.
    - For each chunk:
        * Encodes as in-memory WAV
        * Sends to the validation endpoint
        * If valid -> sends to the model endpoint
        * Collects a simplified result:
              {"index": chunk_index, "prediction": ..., "confidence": ...}

    Returns:
        List[dict]: one entry per VALID chunk.
    """
    import io
    import torchaudio

    print("\n" + "=" * 70)
    print("[Router-Upload] BEGIN NON-SPEECH CLASSIFICATION FOR UPLOAD")

    if not isinstance(waveform, torch.Tensor):
        print("[Router-Upload] Converting numpy array → tensor")
        waveform = torch.tensor(waveform)

    # Ensure (channels, samples)
    if waveform.ndim == 1:
        print("[Router-Upload] Input waveform is 1D — reshaping to mono tensor")
        waveform = waveform.unsqueeze(0)

    total_samples = waveform.shape[1]
    print(f"[Router-Upload] Waveform received | shape={tuple(waveform.shape)}, total_samples={total_samples}, sample_rate={sample_rate}")

    if total_samples == 0:
        print("[Router-Upload] WARNING: Empty NON-SPEECH waveform for classification. Returning empty list.\n")
        return []

    max_chunk_samples = int(5 * sample_rate)  # 5-second chunks
    chunk_results = []
    chunk_index = 0
    start = 0

    print(f"[Router-Upload] Splitting NON-SPEECH waveform into <=5 sec chunks | max_chunk_samples={max_chunk_samples}")

    # Main chunk loop
    while start < total_samples:
        end = min(start + max_chunk_samples, total_samples)
        chunk = waveform[:, start:end]

        # Skip zero-length chunks defensively
        if chunk.shape[1] == 0:
            print(f"[Router-Upload] WARNING: Zero-length chunk encountered at index {chunk_index} — skipping.")
            start = end
            chunk_index += 1
            continue

        duration_sec = chunk.shape[1] / float(sample_rate)
        print(
            f"[Router-Upload] Processing NON-SPEECH chunk {chunk_index}\n"
            f"[Router-Upload] samples={chunk.shape[1]}, duration={duration_sec:.3f}s"
        )

        # Encode chunk → WAV bytes
        buffer = io.BytesIO()
        print(f"[Router-Upload] Encoding chunk {chunk_index} to WAV bytes...")
        try:
            torchaudio.save(buffer, chunk, sample_rate, format="wav")
            buffer.seek(0)
            print(f"[Router-Upload] Encoding complete for chunk {chunk_index}")
        except Exception as e:
            print(f"[Router-Upload] ERROR encoding upload chunk {chunk_index} to WAV: {e}")
            start = end
            chunk_index += 1
            continue

        # VALIDATION ENDPOINT CALL
        print(f"[Router-Upload] Sending chunk {chunk_index} to VALIDATION endpoint...")
        try:
            val_resp = requests.post(
                VALIDATION_URL,
                files={"audio": ("upload_nonspeech.wav", buffer, "audio/wav")},
                data={
                    "recording_id": "upload",
                    "clip_index": chunk_index,
                },
            )
            val_json = val_resp.json()
            print(f"[Router-Upload] Validation response for chunk {chunk_index}: {val_json}")
        except Exception as e:
            print(f"[Router-Upload] ERROR contacting validation endpoint for chunk {chunk_index}: {e}")
            start = end
            chunk_index += 1
            continue

        if not val_json.get("valid", False):
            print(
                f"[Router-Upload] Validation FAILED for NON-SPEECH chunk {chunk_index}. "
                f"Skipping model classification."
            )
            start = end
            chunk_index += 1
            continue

        print(f"[Router-Upload] Validation PASSED for chunk {chunk_index} — sending to MODEL endpoint.")

        # MODEL ENDPOINT CALL
        try:
            buffer.seek(0)
            print(f"[Router-Upload] Sending chunk {chunk_index} to CNN model endpoint...")
            model_resp = requests.post(
                MODEL_URL,
                files={"audio": ("upload_nonspeech.wav", buffer, "audio/wav")},
                data={
                    "recording_id": "upload",
                    "clip_index": chunk_index,
                    "is_last_clip": "false",
                },
            )
            model_json = model_resp.json()
            print(f"[Router-Upload] Model endpoint response for chunk {chunk_index}: {model_json}")
        except Exception as e:
            print(f"[Router-Upload] ERROR contacting model endpoint for chunk {chunk_index}: {e}")
            start = end
            chunk_index += 1
            continue

        # Extract prediction + confidence
        prediction = model_json.get("prediction")
        confidence = model_json.get("confidence")

        # Store result
        chunk_results.append({
            "index": chunk_index,
            "prediction": prediction,
            "confidence": confidence,
        })

        print(
            f"[Router-Upload] Model result stored for chunk {chunk_index} | "
            f"prediction={prediction}, confidence={confidence}"
        )

        # Advance chunk window
        start = end
        chunk_index += 1

    # FINAL SUMMARY LOG
    print(
        f"[Router-Upload] Completed NON-SPEECH classification for upload | "
        f"valid_chunks={len(chunk_results)} | total_chunks={chunk_index}"
    )
    print("=" * 70 + "\n")

    return chunk_results


def run_gemini_for_upload(recording_id: str, transcription_text: str, nonspeech_results: list):
    """
    Run Gemini analysis for the upload workflow, WITHOUT SessionManager.
    """

    # Lazy import
    from routes.gemini_analysis import _run_gemini_analysis

    print("\n" + "=" * 70)
    print("[Gemini-Upload] BEGIN GEMINI ANALYSIS (UPLOAD WORKFLOW)")
    print(f"[Gemini-Upload] recording_id={recording_id}")

    # Readiness / Context Flags 
    transcription_completed = transcription_text is not None
    has_transcription_text = bool(transcription_text and str(transcription_text).strip())
    has_model_results = bool(nonspeech_results)
    finished = True  # Upload analysis is always "finished" immediately

    print("[Gemini-Upload] CONTEXT FLAGS:")
    print(f"[Gemini-Upload] transcription_completed={transcription_completed}")
    print(f"[Gemini-Upload] has_transcription_text={has_transcription_text}")
    print(f"[Gemini-Upload] has_model_results={has_model_results}")
    print(f"[Gemini-Upload] finished={finished}")

    # ------------------------------------------------------------
    # ZERO-CONTEXT CASE: matches Level 1 behavior in gemini_check_or_analyze
    # ------------------------------------------------------------
    if transcription_completed and not has_transcription_text and not has_model_results:
        print(
            f"[Gemini-Upload] NOT ENOUGH CONTEXT | recording_id={recording_id} | "
            f"empty transcript AND no model results."
        )
        print("=" * 70 + "\n")
        return {
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
        }

    # CNN RESULTS OBJECT PREP
    if not nonspeech_results:
        print("[Gemini-Upload] No CNN results detected — constructing default none_detected entry.")
        cnn_results_obj = [{
            "class": "none_detected",
            "confidence": 0.0,
            "note": (
                "No environmental sounds passed validation; the audio was primarily "
                "speech or silence, so the environmental sound model did not produce "
                "any detections."
            ),
        }]
        num_model_results = 0
    else:
        print(f"[Gemini-Upload] Received {len(nonspeech_results)} CNN detection(s).")
        cnn_results_obj = nonspeech_results
        num_model_results = len(nonspeech_results)

    # PRE-GEMINI INVOCATION LOGS 
    transcript_len = len(transcription_text or "")
    print("[Gemini-Upload] Preparing to invoke Gemini analysis...")
    print(f"[Gemini-Upload] Transcript length={transcript_len} characters")
    print(f"[Gemini-Upload] CNN result count={num_model_results}")

    try:
        # RUN GEMINI (internally logs cleaned output)
        analysis = _run_gemini_analysis(
            transcription_text=transcription_text,
            cnn_results_obj=cnn_results_obj,
        )

        # Extract confidence fields
        confidence_score = analysis.get("confidence_score")
        confidence_reasoning = analysis.get("confidence_reasoning")

        low_confidence = None
        if isinstance(confidence_score, (int, float)):
            try:
                score_value = float(confidence_score)
                low_confidence = score_value < 0.4
            except (TypeError, ValueError):
                low_confidence = None

        # POST-GEMINI LOGS 
        print(
            f"[Gemini-Upload] Gemini analysis complete | recording_id={recording_id}"
        )
        print(
            f"[Gemini-Upload]   risk_score={analysis.get('risk_score')} | "
            f"benefit_score={analysis.get('benefit_score')}"
        )
        print(
            f"[Gemini-Upload]   confidence_score={confidence_score} | "
            f"low_confidence={low_confidence}"
        )

        # Format CNN detections same as live workflow
        if nonspeech_results:
            formatted_detections = [
                {
                    "class": det.get("prediction"),
                    "confidence": det.get("confidence"),
                }
                for det in nonspeech_results
            ]
        else:
            formatted_detections = []

        analysis["model_results"] = formatted_detections

        print("[Gemini-Upload] Gemini analysis fully completed for upload workflow")
        print("=" * 70 + "\n")

        # Return final payload
        return {
            "recording_id": recording_id,
            "status": "completed",
            "transcription_completed": transcription_completed,
            "has_transcription": has_transcription_text,
            "has_model_results": has_model_results,
            "num_model_results": num_model_results,
            "finished": finished,
            "analysis": analysis,
            "confidence_score": confidence_score,
            "confidence_reasoning": confidence_reasoning,
            "low_confidence": low_confidence,
        }

    except Exception as e:
        print(
            f"[Gemini-Upload] ERROR during analysis | recording_id={recording_id} | error={e}"
        )
        print("=" * 70 + "\n")
        return {
            "recording_id": recording_id,
            "status": "gemini_error",
            "transcription_completed": transcription_completed,
            "has_transcription": has_transcription_text,
            "has_model_results": has_model_results,
            "num_model_results": len(nonspeech_results),
            "finished": finished,
            "error": str(e),
        }
