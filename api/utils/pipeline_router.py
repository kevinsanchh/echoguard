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