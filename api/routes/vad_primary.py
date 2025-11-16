from flask import Blueprint, request, jsonify, current_app
from utils.audio_utils import load_audio
from utils.vad_utils import (
    run_vad_on_waveform,
    extract_speech_segments,
    extract_nonspeech_segments,
    stitch_segments,
)
import tempfile
from pathlib import Path
import os
import torch  # NEW: for waveform stats / normalization


vad_bp = Blueprint("vad", __name__, url_prefix="/process")


@vad_bp.route("/vad", methods=["POST"])
def process_vad():
    # =========================================================
    # 1. Basic request validation
    # =========================================================
    if "audio" not in request.files:
        print("\n[VAD] ERROR: No 'audio' file part in the request.\n")
        return jsonify({"error": "No audio file part in the request"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        print("\n[VAD] ERROR: No selected file in the request.\n")
        return jsonify({"error": "No selected file"}), 400

    # Tracking fields (same pattern as /api/audio-upload)
    recording_id = request.form.get("recording_id", type=str)
    clip_index = request.form.get("clip_index", type=int)
    is_last_clip_raw = request.form.get("is_last_clip")

    if recording_id is None or clip_index is None or is_last_clip_raw is None:
        print("\n[VAD] ERROR: Missing one or more required tracking fields.\n")
        return jsonify({
            "error": "Missing required fields: recording_id, clip_index, and is_last_clip are required."
        }), 400

    is_last_clip = is_last_clip_raw.lower() == "true"

    # Router imports
    from utils.pipeline_router import (
        store_speech_segment,
        store_nonspeech_segment,
        route_non_speech_for_classification,
        send_speech_to_transcription,
    )

    temp_path = None

    try:
        # =====================================================
        # 2. Save temp WAV file
        # =====================================================
        instance_path = current_app.instance_path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=instance_path) as tmp:
            audio_file.save(tmp.name)
            temp_path = Path(tmp.name)

        print(
            "\n" + "=" * 70 +
            f"\n[VAD] New clip received | recording_id={recording_id} | clip_index={clip_index}"
            f"\n[VAD] Saved incoming WAV file '{audio_file.filename}' to {temp_path}"
        )

        # =====================================================
        # 3. Load waveform
        # =====================================================
        waveform = load_audio(temp_path)  # expected shape [1, num_samples]

        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)

        # Ensure 2D [1, num_samples]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # -------------------------
        # 3a. Waveform diagnostics
        # -------------------------
        with torch.no_grad():
            max_abs = waveform.abs().max().item()
            mean_val = waveform.mean().item()
            rms = torch.sqrt(torch.mean(waveform ** 2)).item()

        print(
            f"[VAD] Waveform stats for clip {clip_index} | "
            f"shape={tuple(waveform.shape)}, max_abs={max_abs:.6f}, "
            f"mean={mean_val:.6f}, rms={rms:.6f}"
        )

        # -------------------------
        # 3b. Normalize / boost level for VAD ONLY
        # -------------------------
        # If the signal is extremely quiet but non-zero, scale it up so
        # the VAD model sees a healthy dynamic range.
        #
        # NOTE: this does NOT change what is stored in session_manager,
        # and it does NOT affect the CNN model later. It only affects
        # the VAD step.
        if max_abs > 1e-4:
            target_peak = 0.9  # keep a little headroom
            gain = target_peak / max_abs
            waveform = (waveform * gain).clamp(-1.0, 1.0)

            with torch.no_grad():
                new_max = waveform.abs().max().item()
                new_rms = torch.sqrt(torch.mean(waveform ** 2)).item()

            print(
                f"[VAD] Applied normalization for clip {clip_index} | "
                f"gain={gain:.2f}, new_max_abs={new_max:.6f}, new_rms={new_rms:.6f}"
            )
        else:
            print(
                f"[VAD] Waveform for clip {clip_index} is extremely quiet (max_abs={max_abs:.6f}); "
                f"skipping normalization."
            )

        # =====================================================
        # 4. Access VAD model + helpers
        # =====================================================
        vad_model = current_app.config["vad_model"]
        vad_helpers = current_app.config["vad_helpers"]
        sample_rate = current_app.config.get("sample_rate", 16000)

        # =====================================================
        # 5. Run VAD
        # =====================================================
        speech_ts = run_vad_on_waveform(
            waveform=waveform,
            model=vad_model,
            vad_helpers=vad_helpers,
            sample_rate=sample_rate,
            threshold=0.25,
            min_speech_duration_ms=80,
            min_silence_duration_ms=0,
        )

        print(f"[VAD] Raw speech_ts for clip {clip_index}: {speech_ts}")
        speech_detected = len(speech_ts) > 0
        print(
            f"[VAD] VAD result for clip {clip_index} | "
            f"speech_detected={speech_detected} | speech_regions={len(speech_ts)}"
        )

        # Detailed per-segment logging
        for i, seg in enumerate(speech_ts):
            start_s = seg["start"] / sample_rate
            end_s   = seg["end"]   / sample_rate
            dur_s   = end_s - start_s
            print(f"[VAD] Speech region {i}: start={start_s:.2f}s, end={end_s:.2f}s, duration={dur_s:.2f}s")

        # =====================================================
        # 6. Extract segments
        # =====================================================
        speech_segments = extract_speech_segments(waveform, speech_ts)
        nonspeech_segments = extract_nonspeech_segments(waveform, speech_ts)

        print(f"[VAD] Extracted {len(speech_segments)} speech segments for clip {clip_index}")
        print(f"[VAD] Extracted {len(nonspeech_segments)} non-speech segments for clip {clip_index}")

        print(
            f"[VAD] Segments summary | recording_id={recording_id}, clip_index={clip_index}, "
            f"num_speech_segments={len(speech_segments)}, "
            f"num_nonspeech_segments={len(nonspeech_segments)}"
        )

        # =====================================================
        # 7. Stitch non-speech segments
        # =====================================================
        stitched_nonspeech = stitch_segments(nonspeech_segments)

        if stitched_nonspeech is not None:
            print(
                f"[VAD] Stitched NON-SPEECH segment for clip {clip_index} | "
                f"waveform shape={stitched_nonspeech.shape}"
            )
        else:
            print(f"[VAD] No NON-SPEECH audio to stitch for clip {clip_index}.")

        # =====================================================
        # 8. ROUTE NON-SPEECH → validate → model → store result
        # =====================================================
        if stitched_nonspeech is not None:
            # Notify router (does NOT store waveform)
            store_nonspeech_segment(recording_id, clip_index)

            # Send waveform onward through pipeline
            classification_result = route_non_speech_for_classification(
                recording_id=recording_id,
                clip_index=clip_index,
                waveform=stitched_nonspeech,
                is_last_clip=is_last_clip,
            )

            print(
                f"[VAD] Routed NON-SPEECH for classification | "
                f"recording={recording_id}, clip={clip_index}, "
                f"classification_result={classification_result}"
            )
        else:
            print(f"[VAD] No NON-SPEECH detected for clip {clip_index}; skipping classification routing.")

        # =====================================================
        # 9. STORE SPEECH SEGMENTS for later transcription
        # =====================================================
        if len(speech_segments) > 0:
            store_speech_segment(recording_id, clip_index, speech_segments)
            print(f"[VAD] Confirm: stored {len(speech_segments)} speech segment(s) for clip {clip_index}")
        else:
            print(f"[VAD] No SPEECH detected for clip {clip_index} (after VAD).")

        # =====================================================
        # 10. LAST CLIP → trigger transcription
        # =====================================================
        if is_last_clip:
            print(f"[VAD] LAST CLIP for recording {recording_id}. Triggering transcription...")
            send_speech_to_transcription(recording_id)

        # =====================================================
        # 11. Debug JSON response (TEMP ONLY)
        # =====================================================
        print("=" * 70 + "\n")
        return jsonify({
            "message": "VAD processing completed for this clip.",
            "recording_id": recording_id,
            "clip_index": clip_index,
            "speech_detected": speech_detected,
            "num_speech_segments": len(speech_segments),
            "num_nonspeech_segments": len(nonspeech_segments),
        }), 200

    except Exception as e:
        print(f"\n[VAD] ERROR: Failed to process VAD: {e}\n")
        return jsonify({"error": f"VAD processing failed: {str(e)}"}), 500

    finally:
        if temp_path and temp_path.exists():
            os.remove(temp_path)
            print(f"[VAD] DEBUG: Cleaned up temp WAV file: {temp_path}")
