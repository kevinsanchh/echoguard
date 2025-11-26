# routes/upload_audio.py

from flask import Blueprint, request, jsonify, current_app
from utils.audio_utils import load_audio
from utils.vad_utils import (
    run_vad_on_waveform,
    extract_speech_segments,
    extract_nonspeech_segments,
    stitch_segments,
)
from utils.pipeline_router import (
    classify_nonspeech_for_upload,
    run_gemini_for_upload,
)
import tempfile
from pathlib import Path
import os
import torch
import torchaudio
from utils.audio_utils import resample_to_16k
import uuid  # needed for synthetic recording IDs

upload_bp = Blueprint("upload", __name__, url_prefix="/process")


@upload_bp.route("/upload-audio", methods=["POST"])
def upload_audio_analysis():
    """
    Upload-based audio pipeline:
    - Load file
    - NORMALIZE waveform (for VAD only)
    - Run VAD on normalized
    - Extract speech / NON-speech from RAW waveform
    - Stitch NON-speech (using RAW waveform)
    - Validate + classify NON-speech (RAW only)
    - Transcribe full RAW waveform (unchanged)
    - Run Gemini
    """

    # 1. Validate request
    if "audio" not in request.files:
        print("\n[Upload] ERROR: No 'audio' file in request.\n")
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        print("\n[Upload] ERROR: Empty filename for audio upload.\n")
        return jsonify({"error": "No audio filename provided"}), 400

    recording_id = request.form.get("recording_id", type=str)

    if recording_id is None:
        recording_id = f"upload-{uuid.uuid4()}"
        print(f"[Upload] No recording_id provided — generated synthetic ID: {recording_id}")
    else:
        print(f"[Upload] Using provided recording_id: {recording_id}")

    print(
        "\n" + "=" * 70 +
        f"\n[Upload] New uploaded audio received | recording_id={recording_id}"
    )

    temp_path = None
    final_path = None

    try:
        instance_path = current_app.instance_path

        # 2. Save uploaded WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=instance_path) as tmp:
            audio_file.save(tmp.name)
            temp_path = Path(tmp.name)

        final_path = Path(instance_path) / f"uploaded_{recording_id}.wav"
        os.replace(temp_path, final_path)
        temp_path = final_path

        print(f"[Upload] Saved uploaded WAV file to {final_path}")

        # 3. Load RAW waveform
        raw_waveform = load_audio(final_path)

        if not isinstance(raw_waveform, torch.Tensor):
            raw_waveform = torch.tensor(raw_waveform)

        if raw_waveform.ndim == 1:
            raw_waveform = raw_waveform.unsqueeze(0)

        with torch.no_grad():
            max_abs = raw_waveform.abs().max().item()
            rms = torch.sqrt(torch.mean(raw_waveform ** 2)).item()

        print(
            f"[Upload] Waveform stats (RAW) | "
            f"shape={tuple(raw_waveform.shape)}, max_abs={max_abs:.6f}, rms={rms:.6f}"
        )

        # >>> KEY CHANGE #1: Preserve RAW waveform for segmentation + CNN
        waveform_for_vad = raw_waveform.clone()

        # 4. NORMALIZATION (VAD ONLY)
        if max_abs > 1e-4:
            target_peak = 0.9
            gain = target_peak / max_abs
            waveform_for_vad = (waveform_for_vad * gain).clamp(-1.0, 1.0)

            with torch.no_grad():
                new_max = waveform_for_vad.abs().max().item()
                new_rms = torch.sqrt(torch.mean(waveform_for_vad ** 2)).item()

            print(
                f"[Upload] Applied normalization for VAD | "
                f"gain={gain:.2f}, new_max_abs={new_max:.6f}, new_rms={new_rms:.6f}"
            )
        else:
            print(
                f"[Upload] Waveform extremely quiet (max_abs={max_abs:.6f}); skipping normalization."
            )

        # 5. Retrieve VAD model
        vad_model = current_app.config["vad_model"]
        vad_helpers = current_app.config["vad_helpers"]
        sample_rate = current_app.config.get("sample_rate", 16000)

        # 6. Run VAD on NORMALIZED waveform
        speech_ts = run_vad_on_waveform(
            waveform=waveform_for_vad,
            model=vad_model,
            vad_helpers=vad_helpers,
            sample_rate=sample_rate,
            threshold=0.20,
            min_speech_duration_ms=100,
            min_silence_duration_ms=250,
        )

        print(f"[Upload] VAD speech_ts: {speech_ts}")
        print(f"[Upload] speech_detected={len(speech_ts) > 0}, regions={len(speech_ts)}")

        # 7. Extract segments FROM RAW WAVEFORM (fix)
        speech_segments = extract_speech_segments(raw_waveform, speech_ts)
        nonspeech_segments = extract_nonspeech_segments(raw_waveform, speech_ts)

        print(
            f"[Upload] Extracted segments | speech={len(speech_segments)}, "
            f"nonspeech={len(nonspeech_segments)}"
        )

        # 8. Stitch NON-SPEECH **raw** segments
        stitched_nonspeech = stitch_segments(nonspeech_segments)
        if stitched_nonspeech is not None:
            print(
                f"[Upload] Stitched NON-SPEECH shape={tuple(stitched_nonspeech.shape)}"
            )
        else:
            print("[Upload] No NON-SPEECH detected.")

        # 9. Classify NON-SPEECH using RAW waveforms
        nonspeech_results = None
        if stitched_nonspeech is not None:
            nonspeech_results = classify_nonspeech_for_upload(
                stitched_nonspeech,
                sample_rate=sample_rate,
            )
        else:
            print("[Upload] Skipping NON-SPEECH classification (no segments).")

        # 10. Transcribe full RAW waveform (unchanged behavior)
        transcription_text, transcription_segments = _transcribe_uploaded_file(
            recording_id,
            raw_waveform,
            sample_rate
        )

        # 11. Run Gemini
        final_result = run_gemini_for_upload(
            recording_id=recording_id,
            transcription_text=transcription_text,
            nonspeech_results=nonspeech_results
        )

        print("=" * 70)
        return final_result, 200

    except Exception as e:
        print(f"\n[Upload] ERROR: {e}\n")
        return jsonify({"error": f"Upload processing failed: {str(e)}"}), 500


def _transcribe_uploaded_file(recording_id, waveform, sample_rate):
    """
    Unified transcription for upload workflow using the GLOBAL Whisper model.
    No per-request loading.
    """

    try:
        # Resample waveform to 16k
        waveform_16k, new_sr = resample_to_16k(
            waveform=waveform,
            orig_sr=sample_rate,
            target_sr=16000
        )

        model = current_app.config["whisper_model"]
        device = "cpu"  # logging consistency

        audio_np = waveform_16k.squeeze(0).numpy()

        print(
            f"[Upload] Starting Faster-Whisper transcription | "
            f"recording_id={recording_id}, device={device}"
        )

        segments, info = model.transcribe(
            audio_np,
            beam_size=5,
            task="transcribe"
        )

        segments = list(segments)

        text_parts = []
        segment_dicts = []

        for seg in segments:
            text = seg.text or ""
            text_parts.append(text)

            segment_dicts.append({
                "start": seg.start,
                "end": seg.end,
                "text": text.strip()
            })

        full_text = " ".join(t.strip() for t in text_parts if t.strip())

        print(
            f"[Upload] Transcription completed | recording_id={recording_id}, "
            f"language={getattr(info, 'language', None)}, "
            f"num_segments={len(segment_dicts)}"
        )

        print(
            f"[Upload-Transcribe] TRANSCRIPT TEXT (upload):\n{full_text}\n"
            f"[Upload-Transcribe] NUM SEGMENTS: {len(segment_dicts)}"
        )

        return full_text, segment_dicts

    except Exception as e:
        print(
            f"[Upload] ERROR: Whisper transcription failed | "
            f"recording_id={recording_id} | error={e}"
        )
        return "", []