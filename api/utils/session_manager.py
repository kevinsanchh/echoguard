import os

# -----------------------------------------------------------
# GLOBAL IN-MEMORY STORE
# -----------------------------------------------------------

# Each session now stores:
# - nonspeech_results: predictions for stitched non-speech segments
# - speech_segments: raw speech waveforms (per clip)
# - full_clips: stored WAV file paths for full 5-second chunks
# - transcription: text + segment timestamps (added)
# - finished: whether last clip was processed
session_recordings = {}


# -----------------------------------------------------------
# INTERNAL HELPER
# -----------------------------------------------------------

def _ensure_session(recording_id):
    """
    Internal helper to ensure a session structure exists for a given recording_id.
    This centralizes the default structure so we don't duplicate it.
    """
    if recording_id not in session_recordings:
        session_recordings[recording_id] = {
            "nonspeech_results": [],
            "speech_segments": {},
            "full_clips": {},
            "finished": False,
            "transcription": {          # <-- ADDED
                "text": None,
                "segments": []
            },
        }


# -----------------------------------------------------------
# NON-SPEECH RESULT STORAGE (model outputs)
# -----------------------------------------------------------

def add_nonspeech_result(recording_id, clip_index, prediction, confidence, is_last_clip=False):
    _ensure_session(recording_id)

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
    _ensure_session(recording_id)

    session_recordings[recording_id]["speech_segments"][clip_index] = segments_list

    print(
        f"[SessionManager] Stored {len(segments_list)} SPEECH segments "
        f"for recording {recording_id}, clip {clip_index}"
    )


# -----------------------------------------------------------
# FULL CLIP STORAGE (raw 5-sec WAVs)
# -----------------------------------------------------------

def add_full_clip(recording_id, clip_index, file_path):
    """
    Store the path to a full 5-second RAW WAV file for this recording/clip.
    """
    _ensure_session(recording_id)

    session_recordings[recording_id]["full_clips"][clip_index] = str(file_path)


def get_all_full_clips(recording_id):
    """
    Return a list of full-clip file paths ordered by clip_index.
    """
    session = session_recordings.get(recording_id)
    if not session:
        return []

    full_clips = session.get("full_clips", {})
    ordered_paths = []

    for clip_idx in sorted(full_clips.keys()):
        ordered_paths.append(full_clips[clip_idx])

    return ordered_paths


def delete_all_full_clips(recording_id):
    """
    After transcription completes, delete all WAV files and clear memory.
    """
    session = session_recordings.get(recording_id)
    if not session:
        print(f"[SessionManager] delete_all_full_clips: No session for recording {recording_id}.")
        return

    full_clips = session.get("full_clips", {})

    for clip_idx, path in full_clips.items():
        try:
            if path and os.path.exists(path):
                os.remove(path)
                print(
                    f"[SessionManager] Deleted FULL CLIP file for recording {recording_id}, "
                    f"clip {clip_idx} | path={path}"
                )
            else:
                print(
                    f"[SessionManager] FULL CLIP file not found for recording {recording_id}, "
                    f"clip {clip_idx} | path={path}"
                )
        except Exception as e:
            print(
                f"[SessionManager] ERROR deleting FULL CLIP file for recording {recording_id}, "
                f"clip {clip_idx} | path={path} | error={e}"
            )

    session_recordings[recording_id]["full_clips"] = {}
    print(f"[SessionManager] Cleared FULL CLIP paths for recording {recording_id}.")


# -----------------------------------------------------------
# LEGACY WRAPPERS
# -----------------------------------------------------------

def add_clip_result(recording_id, clip_index, prediction, confidence, is_last_clip=False):
    add_nonspeech_result(
        recording_id=recording_id,
        clip_index=clip_index,
        prediction=prediction,
        confidence=confidence,
        is_last_clip=is_last_clip
    )


def finish_session(recording_id):
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


# -----------------------------------------------------------
# NEW — TRANSCRIPTION STORAGE
# -----------------------------------------------------------

def store_transcription(recording_id, text, segments):
    """
    Stores the final Faster-Whisper transcription for a recording.
    Does NOT delete or modify any existing stored session data.
    """
    if recording_id not in session_recordings:
        print(f"[SessionManager] ERROR: Tried to store transcription for missing recording_id={recording_id}")
        return

    session_recordings[recording_id]["transcription"]["text"] = text
    session_recordings[recording_id]["transcription"]["segments"] = segments

    print(
        f"[SessionManager] Stored TRANSCRIPTION | recording_id={recording_id} | "
        f"chars={len(text)} | segments={len(segments)}"
    )
