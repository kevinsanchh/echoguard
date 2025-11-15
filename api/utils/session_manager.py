# -----------------------------------------------------------
# GLOBAL IN-MEMORY STORE
# -----------------------------------------------------------

# Each session now stores:
# - nonspeech_results: predictions for stitched non-speech segments
# - speech_segments: raw speech waveforms (per clip)
# - finished: whether last clip was processed
session_recordings = {}


# -----------------------------------------------------------
# NON-SPEECH RESULT STORAGE (model outputs)
# -----------------------------------------------------------

def add_nonspeech_result(recording_id, clip_index, prediction, confidence, is_last_clip=False):
    if recording_id not in session_recordings:
        session_recordings[recording_id] = {
            "nonspeech_results": [],
            "speech_segments": {},
            "finished": False
        }

    session_recordings[recording_id]["nonspeech_results"].append({
        "index": clip_index,
        "prediction": prediction,
        "confidence": confidence
    })

    print(
        f"[SessionManager] Stored NON-SPEECH result for recording {recording_id}, "
        f"clip {clip_index} | prediction={prediction}, conf={confidence}"
    )

    if is_last_clip:
        session_recordings[recording_id]["finished"] = True
        print(f"[SessionManager] Session {recording_id} marked FINISHED (last clip).")


# -----------------------------------------------------------
# SPEECH SEGMENT STORAGE (raw tensors for transcription)
# -----------------------------------------------------------

def add_speech_segments(recording_id, clip_index, segments_list):
    if recording_id not in session_recordings:
        session_recordings[recording_id] = {
            "nonspeech_results": [],
            "speech_segments": {},
            "finished": False
        }

    session_recordings[recording_id]["speech_segments"][clip_index] = segments_list

    print(
        f"[SessionManager] Stored {len(segments_list)} SPEECH segments "
        f"for recording {recording_id}, clip {clip_index}"
    )


# -----------------------------------------------------------
# TEMPORARY WRAPPERS — Legacy Support for /api/audio-upload
# -----------------------------------------------------------

def add_clip_result(recording_id, clip_index, prediction, confidence, is_last_clip=False):
    """
    TEMPORARY:
    Wrapper for legacy /api/audio-upload endpoint.
    Internally maps to add_nonspeech_result so nothing breaks.
    """
    add_nonspeech_result(
        recording_id=recording_id,
        clip_index=clip_index,
        prediction=prediction,
        confidence=confidence,
        is_last_clip=is_last_clip
    )


def finish_session(recording_id):
    """
    TEMPORARY:
    Legacy wrapper so the old endpoint can still mark the session finished.
    """
    mark_session_finished(recording_id)


# -----------------------------------------------------------
# RETRIEVAL HELPERS
# -----------------------------------------------------------

def get_all_speech_segments(recording_id):
    session = session_recordings.get(recording_id)
    if not session:
        return []

    speech_dict = session.get("speech_segments", {})
    stitched_order = []

    for clip_idx in sorted(speech_dict.keys()):
        stitched_order.extend(speech_dict[clip_idx])

    return stitched_order


def mark_session_finished(recording_id):
    if recording_id in session_recordings:
        session_recordings[recording_id]["finished"] = True
        print(f"[SessionManager] Session {recording_id} marked as finished.")
    else:
        print(f"[SessionManager] Tried to finish unknown session {recording_id}.")


def get_session(recording_id):
    return session_recordings.get(recording_id)


def get_all_sessions():
    return session_recordings