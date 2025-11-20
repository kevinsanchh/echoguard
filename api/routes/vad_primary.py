from flask import Blueprint, request, jsonify, current_app
from utils.audio_utils import load_audio
from utils.vad_utils import (
    run_vad_on_waveform,
    extract_speech_segments,
    extract_nonspeech_segments,
    stitch_segments,
)
from utils.pipeline_router import (
    send_cnn_model_result_to_frontend,
)
import tempfile
from pathlib import Path
import os
import torch  

vad_bp = Blueprint("vad", __name__, url_prefix="/process")


@vad_bp.route("/vad", methods=["POST"])
def process_vad():

    # 1. Basic request validation
    if "audio" not in request.files:
        print("\n[VAD] ERROR: No 'audio' file part in the request.\n")
        return jsonify({"error": "No audio file part in the request"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        print("\n[VAD] ERROR: No selected file in the request.\n")
        return jsonify({"error": "No selected file"}), 400

    recording_id = request.form.get("recording_id", type=str)
    clip_index = request.form.get("clip_index", type=int)
    is_last_clip_raw = request.form.get("is_last_clip")

    if recording_id is None or clip_index is None or is_last_clip_raw is None:
        print("\n[VAD] ERROR: Missing required tracking fields.\n")
        return jsonify({
            "error": "Missing required fields: recording_id, clip_index, and is_last_clip are required."
        }), 400

    is_last_clip = is_last_clip_raw.lower() == "true"

    from utils.pipeline_router import (
        store_speech_segment,
        store_nonspeech_segment,
        route_non_speech_for_classification,
        send_full_clips_to_transcription,
    )

    from utils.session_manager import add_full_clip

    temp_path = None
    final_clip_path = None

    try:
        # 2. Save temp WAV file
        instance_path = current_app.instance_path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=instance_path) as tmp:
            audio_file.save(tmp.name)
            temp_path = Path(tmp.name)

        print(
            "\n" + "=" * 70 +
            f"\n[VAD] New clip received | recording_id={recording_id} | clip_index={clip_index}"
            f"\n[VAD] Saved incoming WAV file '{audio_file.filename}' to {temp_path}"
        )


        final_clip_path = Path(instance_path) / f"recording_{recording_id}_clip_{clip_index}.wav"
        os.replace(temp_path, final_clip_path)
        temp_path = final_clip_path  # update reference to the new name

        add_full_clip(recording_id, clip_index, str(final_clip_path))

        print(
            f"[VAD] Stored full WAV clip for transcription | "
            f"recording_id={recording_id} | clip_index={clip_index} | path={final_clip_path}"
        )


        # 3. Load waveform
        waveform = load_audio(final_clip_path)

        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        with torch.no_grad():
            max_abs = waveform.abs().max().item()
            mean_val = waveform.mean().item()
            rms = torch.sqrt(torch.mean(waveform ** 2)).item()

        print(
            f"[VAD] Waveform stats for clip {clip_index} | "
            f"shape={tuple(waveform.shape)}, max_abs={max_abs:.6f}, "
            f"mean={mean_val:.6f}, rms={rms:.6f}"
        )

        # NORMALIZATION
        if max_abs > 1e-4:
            target_peak = 0.9
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
                f"[VAD] Waveform for clip {clip_index} is extremely quiet "
                f"(max_abs={max_abs:.6f}); skipping normalization."
            )

        # 4. Access VAD model + helpers
        vad_model = current_app.config["vad_model"]
        vad_helpers = current_app.config["vad_helpers"]
        sample_rate = current_app.config.get("sample_rate", 16000)

        # 5. Run VAD
        speech_ts = run_vad_on_waveform(
            waveform=waveform,
            model=vad_model,
            vad_helpers=vad_helpers,
            sample_rate=sample_rate,
            threshold=0.20,
            min_speech_duration_ms=100,
            min_silence_duration_ms=250,
        )

        print(f"[VAD] Raw speech_ts for clip {clip_index}: {speech_ts}")
        speech_detected = len(speech_ts) > 0
        print(
            f"[VAD] VAD result for clip {clip_index} | "
            f"speech_detected={speech_detected} | speech_regions={len(speech_ts)}"
        )

        for i, seg in enumerate(speech_ts):
            start_s = seg["start"] / sample_rate
            end_s   = seg["end"]   / sample_rate
            dur_s   = end_s - start_s
            print(f"[VAD] Speech region {i}: start={start_s:.2f}s, end={end_s:.2f}s, duration={dur_s:.2f}s")

        # 6. Extract segments
        speech_segments = extract_speech_segments(waveform, speech_ts)
        nonspeech_segments = extract_nonspeech_segments(waveform, speech_ts)

        print(f"[VAD] Extracted {len(speech_segments)} speech segments for clip {clip_index}")
        print(f"[VAD] Extracted {len(nonspeech_segments)} non-speech segments for clip {clip_index}")

        print(
            f"[VAD] Segments summary | recording_id={recording_id}, clip_index={clip_index}, "
            f"num_speech_segments={len(speech_segments)}, "
            f"num_nonspeech_segments={len(nonspeech_segments)}"
        )

        # 7. Stitch non-speech
        stitched_nonspeech = stitch_segments(nonspeech_segments)

        classification_result = None

        if stitched_nonspeech is not None:
            print(
                f"[VAD] Stitched NON-SPEECH segment for clip {clip_index} | "
                f"waveform shape={stitched_nonspeech.shape}"
            )
        else:
            print(f"[VAD] No NON-SPEECH audio to stitch for clip {clip_index}.")

        # 8. Route non-speech for classification
        if stitched_nonspeech is not None:
            store_nonspeech_segment(recording_id, clip_index)

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

        # 9. Store speech segments (unchanged)
        if len(speech_segments) > 0:
            store_speech_segment(recording_id, clip_index, speech_segments)
            print(f"[VAD] Confirm: stored {len(speech_segments)} speech segment(s) for clip {clip_index}")
        else:
            print(f"[VAD] No SPEECH detected for clip {clip_index} (after VAD).")

        # 10. LAST CLIP
        if is_last_clip:
            print(f"[VAD] LAST CLIP for recording {recording_id}. Triggering transcription...")
            send_full_clips_to_transcription(recording_id)


        # 11. TEMP DEBUG RESPONSE
           # 11. Build response back to frontend (include CNN detections if present)
        response_payload = {
            "message": "VAD processing completed for this clip.",
            "recording_id": recording_id,
            "clip_index": clip_index,
            "speech_detected": speech_detected,
            "num_speech_segments": len(speech_segments),
            "num_nonspeech_segments": len(nonspeech_segments),
        }

           # If we got a classification result from the CNN model, merge it in
        if classification_result is not None:
            cnn_payload = send_cnn_model_result_to_frontend(
                recording_id, clip_index, classification_result
            )
            if cnn_payload is not None:
                response_payload.update(cnn_payload)

        print("=" * 70 + "\n")
        return jsonify(response_payload), 200

    except Exception as e:
        print(f"\n[VAD] ERROR: Failed to process VAD: {e}\n")
        return jsonify({"error": f"VAD processing failed: {str(e)}"}), 500

    finally:
        # >>> IMPORTANT CHANGE <<<
        # We DO NOT delete final_clip_path here anymore.
        #
        # The file persists until AFTER transcription endpoint runs,
        # at which point SessionManager.delete_all_full_clips(recording_id)
        # will remove them all safely.
        pass
