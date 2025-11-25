# routes/upload_audio.py

from flask import Blueprint, request, jsonify, current_app
import tempfile
from pathlib import Path
import os
import uuid

import torch
import torchaudio

from utils.audio_utils import load_audio, resample_to_16k
from utils.vad_utils import (
    run_vad_on_waveform,
    extract_nonspeech_segments,
    stitch_segments,
)
from utils.pipeline_router import (
    classify_nonspeech_for_upload,
    run_gemini_for_upload,
)
from routes.transcribe import _get_whisper_model  # reuse Faster-Whisper loader

upload_bp = Blueprint("upload_audio", __name__, url_prefix="/process")


def _transcribe_uploaded_file(temp_path: Path):
    """
    Transcribe a single uploaded audio file, using the SAME logic and output
    structure as routes/transcribe.py, but without SessionManager.

    Returns:
        full_text: str
    """

    print(f"[Upload-Transcribe] Preparing to transcribe uploaded file | path={temp_path}")

    # Load waveform (same style as transcribe_full_recording)
    waveforms = []
    base_sample_rate = None

    try:
        wav, sr = torchaudio.load(str(temp_path))

        # Convert to mono if multi-channel
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        base_sample_rate = sr
        waveforms.append(wav)

        if not waveforms or base_sample_rate is None:
            print("[Upload-Transcribe] ERROR: No valid audio waveforms loaded for upload file.")
            raise RuntimeError("No valid audio waveforms could be loaded.")

        # For upload, "merged" waveform is just the single file
        merged_waveform = torch.cat(waveforms, dim=1)

        num_samples = merged_waveform.shape[1]
        duration_sec = num_samples / float(base_sample_rate)

        print(
            f"[Upload-Transcribe] Merged waveform for upload transcription | "
            f"sample_rate={base_sample_rate}, total_samples={num_samples}, "
            f"duration={duration_sec:.3f}s"
        )

        orig_sr = base_sample_rate

        merged_waveform_16k, new_sr = resample_to_16k(
            waveform=merged_waveform,
            orig_sr=orig_sr,
            target_sr=16000,
        )

        print(f"[Upload-Transcribe] Resampled merged waveform to 16kHz | new_sr={new_sr}")

    except Exception as e:
        print(
            f"[Upload-Transcribe] ERROR: Failed to load/prepare upload waveform | "
            f"path={temp_path} | error={e}"
        )
        raise

    # Run Faster-Whisper (exact same pattern as transcribe_full_recording)
    try:
        model, device = _get_whisper_model()

        audio_np = merged_waveform_16k.squeeze(0).numpy()

        print(
            f"[Upload-Transcribe] Starting Faster-Whisper transcription for upload | "
            f"device={device}"
        )

        segments, info = model.transcribe(
            audio_np,
            beam_size=5,
            task="transcribe",
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
                "text": text.strip(),
            })

        full_text = " ".join(
            part.strip() for part in full_text_parts if part.strip()
        )

        print(
            f"[Upload-Transcribe] Transcription completed for upload | "
            f"language={getattr(info, 'language', None)}, "
            f"num_segments={len(segment_dicts)}"
        )
        print(
            f"[Upload-Transcribe] TRANSCRIPT TEXT (upload):\n{full_text}\n"
            f"[Upload-Transcribe] NUM SEGMENTS: {len(segment_dicts)}"
        )

        return full_text

    except Exception as e:
        print(
            f"[Upload-Transcribe] ERROR: Faster-Whisper transcription failed for upload | "
            f"error={e}"
        )
        raise


@upload_bp.route("/upload-audio", methods=["POST"])
def upload_audio_analysis():
    """
    Upload-based audio analysis endpoint.

    Workflow:
    1. Accept a single uploaded audio file (no recording_id / clip_index from frontend).
    2. Save it temporarily into the instance folder.
    3. Load waveform using load_audio() and run RNNoise VAD on the entire file.
    4. Extract NON-SPEECH, stitch, and classify via the existing validation + model endpoints.
    5. Transcribe the full uploaded file using the SAME logic as /process/transcribe.
    6. Run Gemini using the SAME prompt/analysis logic, but WITHOUT SessionManager.
    7. Delete the temporary file.
    8. Return a Gemini-style JSON response (with a synthetic recording_id).
    """

    # 1) Basic request validation
    if "audio" not in request.files:
        print("\n[Upload] ERROR: No 'audio' file part in the request.\n")
        return jsonify({"error": "No audio file part in the request"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        print("\n[Upload] ERROR: No selected file in the request.\n")
        return jsonify({"error": "No selected file"}), 400

    instance_path = current_app.instance_path
    Path(instance_path).mkdir(parents=True, exist_ok=True)

    temp_path = None
    synthetic_recording_id = f"upload-{uuid.uuid4()}"

    try:
        # 2) Save uploaded file into instance folder
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=instance_path) as tmp:
            audio_file.save(tmp.name)
            temp_path = Path(tmp.name)

        print(
            "\n" + "=" * 70 +
            f"\n[Upload] New audio file received for upload workflow | "
            f"synthetic_recording_id={synthetic_recording_id}"
            f"\n[Upload] Saved incoming file '{audio_file.filename}' to {temp_path}"
        )

        # 3) Load waveform for VAD + classification using shared audio loader
        waveform = load_audio(temp_path)

        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        with torch.no_grad():
            max_abs = waveform.abs().max().item()
            mean_val = waveform.mean().item()
            rms = torch.sqrt(torch.mean(waveform ** 2)).item()

        print(
            f"[Upload] Waveform stats (for VAD) | "
            f"shape={tuple(waveform.shape)}, max_abs={max_abs:.6f}, "
            f"mean={mean_val:.6f}, rms={rms:.6f}"
        )

        # Normalization (same logic as vad_primary)
        if max_abs > 1e-4:
            target_peak = 0.9
            gain = target_peak / max_abs
            waveform = (waveform * gain).clamp(-1.0, 1.0)

            with torch.no_grad():
                new_max = waveform.abs().max().item()
                new_rms = torch.sqrt(torch.mean(waveform ** 2)).item()

            print(
                f"[Upload] Applied normalization | "
                f"gain={gain:.2f}, new_max_abs={new_max:.6f}, new_rms={new_rms:.6f}"
            )
        else:
            print(
                f"[Upload] Waveform is extremely quiet "
                f"(max_abs={max_abs:.6f}); skipping normalization."
            )

        # 4) Access VAD model + helpers and run RNNoise VAD on entire file
        vad_model = current_app.config["vad_model"]
        vad_helpers = current_app.config["vad_helpers"]
        sample_rate = current_app.config.get("sample_rate", 16000)

        speech_ts = run_vad_on_waveform(
            waveform=waveform,
            model=vad_model,
            vad_helpers=vad_helpers,
            sample_rate=sample_rate,
            threshold=0.20,
            min_speech_duration_ms=100,
            min_silence_duration_ms=250,
        )

        print(f"[Upload] Raw speech_ts (upload): {speech_ts}")
        speech_detected = len(speech_ts) > 0
        print(
            f"[Upload] VAD result (upload) | "
            f"speech_detected={speech_detected} | speech_regions={len(speech_ts)}"
        )

        # 5) Extract NON-SPEECH segments and stitch
        nonspeech_segments = extract_nonspeech_segments(waveform, speech_ts)
        print(f"[Upload] Extracted {len(nonspeech_segments)} NON-SPEECH segments for upload file")

        stitched_nonspeech = stitch_segments(nonspeech_segments)
        upload_nonspeech_results = []

        if stitched_nonspeech is not None:
            print(
                f"[Upload] Stitched NON-SPEECH waveform (upload) | "
                f"shape={tuple(stitched_nonspeech.shape)}"
            )

            # Classify NON-SPEECH using existing validation + model endpoints
            upload_nonspeech_results = classify_nonspeech_for_upload(
                stitched_nonspeech,
                sample_rate=sample_rate,
            )
        else:
            print("[Upload] No NON-SPEECH audio detected; skipping CNN classification.")

        # 6) Transcribe the full uploaded file using the SAME logic as /process/transcribe
        try:
            transcript_text = _transcribe_uploaded_file(temp_path)
        except Exception as e:
            print(f"[Upload] ERROR during transcription of uploaded file: {e}")
            return jsonify({
                "recording_id": synthetic_recording_id,
                "status": "error",
                "message": f"Transcription failed for uploaded audio: {str(e)}",
            }), 500

        # 7) Run Gemini with SAME prompt/analysis behavior, but without SessionManager
        gemini_response = run_gemini_for_upload(
            recording_id=synthetic_recording_id,
            transcription_text=transcript_text,
            nonspeech_results=upload_nonspeech_results,
        )

        print(
            f"[Upload] Gemini upload-analysis response | "
            f"recording_id={gemini_response.get('recording_id')} | "
            f"status={gemini_response.get('status')}"
        )

        print("=" * 70 + "\n")
        return jsonify(gemini_response), 200

    except Exception as e:
        print(f"\n[Upload] ERROR: Failed to process uploaded audio: {e}\n")
        return jsonify({
            "recording_id": synthetic_recording_id,
            "status": "error",
            "message": f"Upload processing failed: {str(e)}",
        }), 500

    finally:
        # 8) Cleanup: delete the temporary file
        if temp_path is not None:
            try:
                if temp_path.exists():
                    os.remove(temp_path)
                    print(f"[Upload] Deleted temporary uploaded file | path={temp_path}")
            except Exception as cleanup_err:
                print(
                    f"[Upload] WARNING: Failed to delete temporary upload file | "
                    f"path={temp_path} | error={cleanup_err}"
                )
