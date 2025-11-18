# routes/transcribe.py

from flask import Blueprint, request, jsonify
from utils.session_manager import (
    get_all_full_clips,
    mark_session_finished,
    delete_all_full_clips,
    store_transcription,
)

import os
from pathlib import Path

import torch
import torchaudio
from faster_whisper import WhisperModel

transcribe_bp = Blueprint("transcribe", __name__, url_prefix="/process")

# --------------------------------------------------------------------
# GLOBAL WHISPER MODEL (lazy-loaded once per process)
# --------------------------------------------------------------------
_WHISPER_MODEL = None
_WHISPER_DEVICE = None


def _get_whisper_model():
    """
    Lazily load the Faster-Whisper model (small) once.
    Uses CUDA if available, otherwise CPU.
    """
    global _WHISPER_MODEL, _WHISPER_DEVICE

    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL, _WHISPER_DEVICE

    device = "cpu"
    compute_type = "int8"

    compute_type = "float16" if device == "cuda" else "int8"

    print(
        f"[Transcribe] Loading Faster-Whisper model 'small' | "
        f"device={device}, compute_type={compute_type}"
    )
    model = WhisperModel("small", device=device, compute_type=compute_type)

    _WHISPER_MODEL = model
    print("[Transcribe] Faster-Whisper model loaded successfully.")

    return _WHISPER_MODEL, _WHISPER_DEVICE


@transcribe_bp.route("/transcribe", methods=["POST"])
def transcribe_full_recording():
    """
    Transcription endpoint for a full recording composed of multiple 5-second clips.

    BEHAVIOR:
    - Expects `recording_id` in form-data.
    - Retrieves all stored FULL CLIP paths from SessionManager.
    - Loads each clip waveform using torchaudio.
    - Ensures consistent sample rate and mono channel.
    - Concatenates all clips into a single waveform.
    - Runs Faster-Whisper (small) to get:
        - full transcript text
        - per-segment timestamps
    - On SUCCESS:
        - Logs summary.
        - (Future step) Stores transcript in SessionManager.
        - Deletes all full clip files for this recording.
        - Marks the session as finished.
        - Returns JSON with transcript and segments.
    - On FAILURE:
        - Logs error.
        - Does NOT delete full clips.
        - Does NOT mark session as finished.
        - Returns JSON with status="error".

    """

    # 1. Extract required field
    recording_id = request.form.get("recording_id", type=str)

    if recording_id is None:
        print("\n[Transcribe] ERROR: Missing 'recording_id' in request.\n")
        return jsonify({"error": "Missing required field 'recording_id'"}), 400

    # 2. Retrieve FULL CLIPS for this recording
    clip_paths = get_all_full_clips(recording_id)

    print(
        f"[Transcribe] Received transcription request | recording_id={recording_id}"
    )

    if not clip_paths:
        print(
            f"[Transcribe] WARNING: No FULL CLIPS found for recording {recording_id}."
        )
        # We still mark as finished (consistent with previous behavior)
        mark_session_finished(recording_id)
        return jsonify({
            "recording_id": recording_id,
            "transcript": "",
            "segments": [],
            "status": "no_clips_found"
        }), 200

    print(
        f"[Transcribe] Found {len(clip_paths)} FULL CLIPS for transcription "
        f"| recording_id={recording_id}"
    )

    # 3. Load and concatenate all WAV clips
    waveforms = []
    base_sample_rate = None

    try:
        for path_str in clip_paths:
            path = Path(path_str)

            if not path.exists():
                print(
                    f"[Transcribe] WARNING: Clip file does not exist and will be skipped | "
                    f"path={path}"
                )
                continue

            wav, sr = torchaudio.load(str(path))

            # Convert to mono if multi-channel
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            if base_sample_rate is None:
                base_sample_rate = sr
            elif sr != base_sample_rate:
                # Simple resample to the base sample rate
                wav = torchaudio.functional.resample(wav, sr, base_sample_rate)

            waveforms.append(wav)

        if not waveforms or base_sample_rate is None:
            print(
                f"[Transcribe] ERROR: Failed to load any valid clip waveforms "
                f"| recording_id={recording_id}"
            )
            return jsonify({
                "recording_id": recording_id,
                "status": "error",
                "message": "No valid audio waveforms could be loaded."
            }), 500

        # Concatenate along time dimension
        merged_waveform = torch.cat(waveforms, dim=1)

        num_samples = merged_waveform.shape[1]
        duration_sec = num_samples / float(base_sample_rate)

        print(
            f"[Transcribe] Merged waveform for transcription | "
            f"recording_id={recording_id}, num_clips={len(waveforms)}, "
            f"sample_rate={base_sample_rate}, total_samples={num_samples}, "
            f"duration={duration_sec:.3f}s"
        )

    except Exception as e:
        print(
            f"[Transcribe] ERROR: Failed to load/merge clip waveforms | "
            f"recording_id={recording_id} | error={e}"
        )
        return jsonify({
            "recording_id": recording_id,
            "status": "error",
            "message": f"Failed to prepare audio for transcription: {str(e)}"
        }), 500

    # 4. Run Faster-Whisper transcription
    try:
        model, device = _get_whisper_model()

        # Convert to numpy 1D array for Faster-Whisper
        audio_np = merged_waveform.squeeze(0).numpy()

        print(
            f"[Transcribe] Starting Faster-Whisper transcription | "
            f"recording_id={recording_id}, device={device}"
        )

        # Basic transcription call (language auto-detection, transcription only)
        segments, info = model.transcribe(
            audio_np,
            beam_size=5,
            task="transcribe"
        )

        # Materialize segments from generator
        segments = list(segments)

        full_text_parts = []
        segment_dicts = []

        for seg in segments:
            text = seg.text or ""
            full_text_parts.append(text)

            segment_dicts.append({
                "start": seg.start,
                "end": seg.end,
                "text": text.strip()
            })

        full_text = " ".join(part.strip() for part in full_text_parts if part.strip())

        print(
            f"[Transcribe] Transcription completed | recording_id={recording_id}, "
            f"language={getattr(info, 'language', None)}, "
            f"num_segments={len(segment_dicts)}"
        )

        store_transcription(
            recording_id=recording_id,
            text=full_text,
            segments=segments
        )

        print(
            f"[Transcribe] Stored transcription | recording_id={recording_id}\n"
            f"[Transcribe] TRANSCRIPT TEXT:\n{full_text}\n"
            f"[Transcribe] NUM SEGMENTS: {len(segments)}"
        )   

    except Exception as e:
        print(
            f"[Transcribe] ERROR: Faster-Whisper transcription failed | "
            f"recording_id={recording_id} | error={e}"
        )

        return jsonify({
            "recording_id": recording_id,
            "status": "error",
            "message": f"Transcription failed: {str(e)}"
        }), 500

    # 5. Cleanup on SUCCESS (delete clips + mark finished)
    try:
        delete_all_full_clips(recording_id)
        print(
            f"[Transcribe] Deleted all FULL CLIP files for recording_id={recording_id}"
        )
    except Exception as cleanup_err:
        # Non-fatal: we still consider transcription successful, but log it.
        print(
            f"[Transcribe] WARNING: Failed to delete FULL CLIP files for "
            f"recording_id={recording_id} | error={cleanup_err}"
        )

    # Mark session finished
    mark_session_finished(recording_id)
    print(
        f"[Transcribe] Transcription pipeline complete for recording_id={recording_id}"
    )

    return jsonify({
        "recording_id": recording_id,
        "transcript": full_text,
        "segments": segment_dicts,
        "status": "ok"
    }), 200