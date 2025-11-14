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

vad_bp = Blueprint("vad", __name__, url_prefix="/process")

@vad_bp.route("/vad", methods=["POST"])
def process_vad():
    
     # 1. Basic request validation
    if "audio" not in request.files:
        print("[VAD] ERROR: No 'audio' file part in the request.")
        return jsonify({"error": "No audio file part in the request"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        print("[VAD] ERROR: No selected file in the request.")
        return jsonify({"error": "No selected file"}), 400


    # Tracking fields (same pattern as /api/audio-upload)
    recording_id = request.form.get("recording_id", type=str)
    clip_index = request.form.get("clip_index", type=int)
    is_last_clip_raw = request.form.get("is_last_clip")

    if recording_id is None or clip_index is None or is_last_clip_raw is None:
        print("[VAD] ERROR: Missing one or more required tracking fields.")
        return jsonify({
            "error": "Missing required fields: recording_id, clip_index, and is_last_clip are required."
        }), 400

    is_last_clip = is_last_clip_raw.lower() == "true"

    temp_path = None

    try:
        
        # 2. Save temp WAV file
        instance_path = current_app.instance_path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=instance_path) as tmp:
            audio_file.save(tmp.name)
            temp_path = Path(tmp.name)

        print(f"[VAD] DEBUG: Received WAV file: {audio_file.filename}. Saved to {temp_path}")

       # 3. Load waveform-
        waveform = load_audio(temp_path)  # shape [1, num_samples]

         # 4. Access VAD model + helpers
        vad_model = current_app.config["vad_model"]
        vad_helpers = current_app.config["vad_helpers"]
        sample_rate = current_app.config.get("sample_rate", 16000)

        # 5. Run VAD on waveform
        speech_ts = run_vad_on_waveform(
            waveform=waveform,
            model=vad_model,
            vad_helpers=vad_helpers,
            sample_rate=sample_rate,
        )

        speech_detected = len(speech_ts) > 0
        print(f"[VAD] DEBUG: speech_detected={speech_detected} | speech_regions={len(speech_ts)}")

         # 6. Extract speech + non-speech segments
        speech_segments = extract_speech_segments(waveform, speech_ts)
        nonspeech_segments = extract_nonspeech_segments(waveform, speech_ts)

        print(
            f"[VAD] DEBUG: recording_id={recording_id}, clip_index={clip_index}, "
            f"num_speech_segments={len(speech_segments)}, num_nonspeech_segments={len(nonspeech_segments)}"
        )

         # 7. Stitch non-speech segments (per 5-second clip)
        stitched_nonspeech = stitch_segments(nonspeech_segments)

        if stitched_nonspeech is not None:
            print(
                f"[VAD] DEBUG: Stitched non-speech segment for clip {clip_index} | "
                f"waveform shape={stitched_nonspeech.shape}"
            )
        else:
            print(f"[VAD] DEBUG: No non-speech audio to stitch for clip {clip_index}.")

        # ---------------------------------------------------------
        # 8. TODO: Route non-speech -> /process/validate-non-speech
        # ---------------------------------------------------------
        # In a future step, this stitched_nonspeech waveform (plus recording_id, clip_index)
        # will be passed to a validation endpoint, which will:
        #   - check that audio is meaningful
        #   - if valid, send it to the classifier (model inference)
        #
        # Example (future pseudocode):
        #   forward_non_speech_to_validation(
        #       recording_id=recording_id,
        #       clip_index=clip_index,
        #       waveform=stitched_nonspeech
        #   )
        #
        # For now, we ONLY log what would happen.

        if stitched_nonspeech is not None:
            print(
                f"[VAD] FUTURE: Would forward NON-SPEECH for recording {recording_id}, "
                f"clip {clip_index} to /process/validate-non-speech"
            )

        # ---------------------------------------------------------
        # 9. TODO: Store speech segments for later transcription
        # ---------------------------------------------------------
        # We will eventually:
        #   - Store these speech_segments in a session-like structure keyed by recording_id
        #   - When is_last_clip is True, gather all speech from all clips
        #   - Then forward them (in order) to /process/transcribe
        #
        # For now, we ONLY log what would happen.

        if len(speech_segments) > 0:
            print(
                f"[VAD] FUTURE: Would store {len(speech_segments)} SPEECH segments for "
                f"recording {recording_id}, clip {clip_index} (to be transcribed later)."
            )

        if is_last_clip:
            print(
                f"[VAD] FUTURE: is_last_clip=True for recording {recording_id}. "
                f"Later, we will trigger transcription of ALL stored speech segments "
                f"and send the final transcript into the Gemini wrapper."
            )

        # ---------------------------------------------------------
        # 10. For now, return a debug-style JSON response
        # ---------------------------------------------------------
        # NOTE: Eventually, this endpoint may not need to return anything directly
        # to the frontend once the full pipeline is wired together. For now, this
        # is extremely useful for testing and development.
        return jsonify({
            "message": "VAD processing completed for this clip (no downstream routing yet).",
            "recording_id": recording_id,
            "clip_index": clip_index,
            "speech_detected": speech_detected,
            "num_speech_segments": len(speech_segments),
            "num_nonspeech_segments": len(nonspeech_segments),
        }), 200

    except Exception as e:
        print(f"[VAD] ERROR: Failed to process VAD: {e}")
        return jsonify({"error": f"VAD processing failed: {str(e)}"}), 500

    finally:
        # Cleanup temp file
        if temp_path and temp_path.exists():
            os.remove(temp_path)
            print(f"[VAD] DEBUG: Cleaned up temp WAV file: {temp_path}")