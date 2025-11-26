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
from utils.audio_utils import resample_to_16k
from utils.pipeline_router import send_to_gemini


transcribe_bp = Blueprint("transcribe", __name__, url_prefix="/process")

# routes/transcribe.py

from flask import Blueprint, request, jsonify, current_app
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
from utils.audio_utils import resample_to_16k
from utils.pipeline_router import send_to_gemini

# No more lazy-loading variables or functions.
# Whisper model is loaded globally in server.py at startup.

transcribe_bp = Blueprint("transcribe", __name__, url_prefix="/process")


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
    - Runs Faster-Whisper (small) preloaded globally in server.py.
    - On SUCCESS:
        - Logs summary.
        - Stores transcript in SessionManager.
        - Deletes all full clip files.
        - Marks the session as finished.
        - Triggers Gemini wrapper.
    - On FAILURE:
        - Logs error.
        - Does NOT delete full clips.
        - Does NOT mark session as finished.
    """

    # 1. Extract required field
    recording_id = request.form.get("recording_id", type=str)

    if recording_id is None:
        print("\n[Transcribe] ERROR: Missing 'recording_id' in request.\n")
        return jsonify({"error": "Missing required field 'recording_id'"}), 400

    # 2. Retrieve FULL CLIPS for this recording
    clip_paths = get_all_full_clips(recording_id)

    print(f"[Transcribe] Received transcription request | recording_id={recording_id}")

    if not clip_paths:
        print(f"[Transcribe] WARNING: No FULL CLIPS found for recording {recording_id}.")
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
                # Resample to the base sample rate
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

        # Resample to 16kHz for Whisper
        merged_waveform_16k, new_sr = resample_to_16k(
            waveform=merged_waveform,
            orig_sr=base_sample_rate,
            target_sr=16000,
        )
        print(f"[Transcribe] Resampled merged waveform to 16kHz | ")

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

    # 4. Run Faster-Whisper transcription (GLOBAL MODEL)
    try:
        model = current_app.config["whisper_model"]
        device = "cpu"  # for logging consistency

        # Convert to numpy array for Faster-Whisper
        audio_np = merged_waveform_16k.squeeze(0).numpy()

        print(
            f"[Transcribe] Starting Faster-Whisper transcription | "
            f"recording_id={recording_id}, device={device}"
        )

        # Transcribe
        segments, info = model.transcribe(
            audio_np,
            beam_size=5,
            task="transcribe"
        )

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

        full_text = " ".join(
            part.strip() for part in full_text_parts if part.strip()
        )

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
        print(
            f"[Transcribe] WARNING: Failed to delete FULL CLIP files for "
            f"recording_id={recording_id} | error={cleanup_err}"
        )

    mark_session_finished(recording_id)
    print(
        f"[Transcribe] Transcription pipeline complete for recording_id={recording_id}"
    )

    # Trigger Gemini wrapper
    try:
        print(f"[Transcribe] Triggering Gemini wrapper | recording_id={recording_id}")
        gemini_result = send_to_gemini(recording_id)

        if gemini_result is not None:
            print(
                f"[Transcribe] Gemini wrapper response | recording_id={recording_id} | "
                f"status={gemini_result.get('status')}"
            )
        else:
            print(
                f"[Transcribe] WARNING: Gemini wrapper returned None "
                f"| recording_id={recording_id}"
            )

    except Exception as gem_err:
        print(
            f"[Transcribe] ERROR: Failed to trigger Gemini wrapper | "
            f"recording_id={recording_id} | error={gem_err}"
        )

    return jsonify({
        "recording_id": recording_id,
        "transcript": full_text,
        "segments": segment_dicts,
        "status": "ok"
    }), 200
