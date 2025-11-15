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
from utils.session_manager import (
    add_speech_segments,
    add_nonspeech_result,
    mark_session_finished,
)

# -----------------------------------------------------------
# CONFIG (these will eventually be loaded from Flask config)
# ----------------------------------------------------------

VALIDATION_URL = Path("http://localhost:8080/process/validate-non-speech")
MODEL_URL = Path("http://localhost:8080/process/model")
TRANSCRIBE_URL = Path("http://localhost:8080/process/transcribe")


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


# -----------------------------------------------------------
# STORE SPEECH SEGMENTS (raw speech)
# -----------------------------------------------------------

def store_speech_segment(recording_id, clip_index, segments):
    """
    Store raw speech segments extracted from VAD.
    """
    add_speech_segments(recording_id, clip_index, segments)
    print(f"[Router] Stored SPEECH segments for recording={recording_id}, clip={clip_index}")


# -----------------------------------------------------------
# ROUTE NON-SPEECH → VALIDATION → MODEL → STORE RESULT
# -----------------------------------------------------------

def route_non_speech_for_classification(recording_id, clip_index, waveform, is_last_clip=False):
    """
    Full pipeline for NON-SPEECH classification:
    1. Send waveform to validation endpoint.
    2. If valid → send to model endpoint.
    3. Store result in session manager.
    4. Return result to frontend.

    NOTE: REAL inference happens in the model endpoint.
    """

    print(f"[Router] Routing NON-SPEECH for classification | recording={recording_id}, clip={clip_index}")

    # ----------------------------------------------
    # Step 1: Validate non-speech data
    # ----------------------------------------------
    try:
        response = requests.post(
            VALIDATION_URL,
            files={"audio": waveform},
            data={"recording_id": recording_id, "clip_index": clip_index}
        )
        response_data = response.json()
    except Exception as e:
        print(f"[Router] ERROR contacting validation endpoint: {e}")
        return None

    if not response_data.get("valid", False):
        print(f"[Router] Validation FAILED for clip {clip_index}. Sending empty detection to FE.")
        add_nonspeech_result(recording_id, clip_index, prediction="none", confidence=0.0, is_last_clip=is_last_clip)
        return {"prediction": "none", "confidence": 0.0}

    # ----------------------------------------------
    # Step 2: Valid → Send to model endpoint
    # ----------------------------------------------
    print(f"[Router] Validation PASSED for clip {clip_index}. Sending to model endpoint.")

    try:
        model_response = requests.post(
            MODEL_URL,
            files={"audio": waveform},
            data={
                "recording_id": recording_id,
                "clip_index": clip_index,
                "is_last_clip": str(is_last_clip)
            }
        )
        model_data = model_response.json()
    except Exception as e:
        print(f"[Router] ERROR contacting model endpoint: {e}")
        return None

    # ----------------------------------------------
    # Step 3: Store result
    # ----------------------------------------------
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


# -----------------------------------------------------------
# ROUTE SPEECH → TRANSCRIPTION
# -----------------------------------------------------------

def send_speech_to_transcription(recording_id):
    """
    Gather all stored speech segments for this recording,
    stitch them inside the transcribe endpoint, and store final transcript.
    """
    print(f"[Router] Routing SPEECH segments for transcription | recording={recording_id}")

    try:
        response = requests.post(
            TRANSCRIBE_URL,
            data={"recording_id": recording_id}
        )
        result = response.json()
    except Exception as e:
        print(f"[Router] ERROR contacting transcription endpoint: {e}")
        return None

    # Mark session finished after transcription
    mark_session_finished(recording_id)

    print(f"[Router] Transcription complete for recording={recording_id}")
    return result