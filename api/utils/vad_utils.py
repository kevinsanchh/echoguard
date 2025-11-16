import torch
import torchaudio
import numpy as np
from pyrnnoise import RNNoise  # Requires: pip install pyrnnoise

# ------------------------------------------------------------
# RNNoise VAD Model Loader (Silero-compatible interface)
# ------------------------------------------------------------

def load_vad_model():
    """
    Load RNNoise-based VAD model.

    Returns:
        model: RNNoise instance
        vad_helpers: dict with configuration (kept for interface compatibility)
    """
    # RNNoise is trained for / expects 48 kHz audio
    target_sample_rate = 48000

    rnnoise_model = RNNoise(sample_rate=target_sample_rate)

    vad_helpers = {
        "target_sample_rate": target_sample_rate,
        "frame_size": 480,  # RNNoise operates on 480-sample frames at 48 kHz (10 ms)
    }

    return rnnoise_model, vad_helpers


# ------------------------------------------------------------
# Run VAD on a waveform (PyTorch tensor) using RNNoise
# ------------------------------------------------------------

def run_vad_on_waveform(
        waveform,
        model,
        vad_helpers,
        sample_rate: int = 16000,
        threshold: float = 0.25,
        min_speech_duration_ms: int = 100,
        min_silence_duration_ms: int = 250,
    ):
    """
    Applies RNNoise-based VAD to a loaded waveform while preserving the
    Silero-style API and output format.

    Parameters:
        waveform (torch.Tensor): Tensor shape [1, num_samples]
        model: RNNoise instance (from pyrnnoise)
        vad_helpers (dict):
            - "target_sample_rate": int (48_000)
            - "frame_size": int (480)
        sample_rate: original audio sample rate used elsewhere in the pipeline
        threshold: speech probability threshold for frame to be considered speech
        min_speech_duration_ms: minimum duration for a speech segment
        min_silence_duration_ms: maximum gap between speech segments to be merged

    Returns:
        speech_timestamps (list[dict]): List of dicts with speech regions:
            [{ "start": int_sample_index, "end": int_sample_index }, ...]
        where indices are in the ORIGINAL sample_rate domain (e.g. 16 kHz),
        matching the expectations of downstream code.
    """

    # --------------------------------------------------------
    # 1. Unpack RNNoise config
    # --------------------------------------------------------
    target_sr = vad_helpers.get("target_sample_rate", 48000)
    frame_size = vad_helpers.get("frame_size", 480)

    # Silero-style: ensure mono [num_samples]
    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)

    # Ensure float tensor on CPU
    waveform = waveform.to(dtype=torch.float32, device="cpu")

    # --------------------------------------------------------
    # 2. Resample to RNNoise target sample rate if needed
    # --------------------------------------------------------
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sr,
        )
        resampled = resampler(waveform.unsqueeze(0)).squeeze(0)
    else:
        resampled = waveform

    # --------------------------------------------------------
    # 3. Convert to int16 numpy [channels, num_samples]
    # --------------------------------------------------------
    # Clamp to [-1, 1], then scale to int16 [-32768, 32767]
    resampled = resampled.clamp(-1.0, 1.0)
    resampled_int16 = (resampled * 32767.0).to(torch.int16).cpu().numpy()

    if resampled_int16.ndim == 1:
        resampled_int16 = resampled_int16.reshape(1, -1)  # [1, num_samples]

    # --------------------------------------------------------
    # 4. Run RNNoise and collect per-frame speech probabilities
    # --------------------------------------------------------
    denoiser = model  # RNNoise instance
    speech_mask = []  # list[bool], one per 10 ms frame

    # pyrnnoise.RNNoise.denoise_chunk yields (speech_probs, denoised_frame)
    for speech_probs, _denoised in denoiser.denoise_chunk(resampled_int16):
        # speech_probs is per-channel; we only have mono
        if isinstance(speech_probs, (list, tuple, np.ndarray)):
            prob = float(speech_probs[0])
        else:
            prob = float(speech_probs)

        is_speech = prob >= threshold
        speech_mask.append(is_speech)

    num_frames = len(speech_mask)
    if num_frames == 0:
        return []

    frame_duration_ms = (frame_size / target_sr) * 1000.0  # ~10 ms

    # --------------------------------------------------------
    # 5. Convert frame-level mask → raw speech segments
    # --------------------------------------------------------
    segments_frames = []
    in_speech = False
    start_frame = 0

    for i, is_speech in enumerate(speech_mask):
        if is_speech and not in_speech:
            in_speech = True
            start_frame = i
        elif not is_speech and in_speech:
            end_frame = i  # exclusive
            segments_frames.append((start_frame, end_frame))
            in_speech = False

    # If we ended in speech, close the last segment
    if in_speech:
        segments_frames.append((start_frame, num_frames))

    if not segments_frames:
        return []

    # --------------------------------------------------------
    # 6. Enforce min_speech_duration_ms
    # --------------------------------------------------------
    min_speech_frames = max(
        1,
        int(np.ceil(min_speech_duration_ms / frame_duration_ms)),
    )

    segments_frames = [
        (s, e) for (s, e) in segments_frames
        if (e - s) >= min_speech_frames
    ]

    if not segments_frames:
        return []

    # --------------------------------------------------------
    # 7. Merge segments separated by short silence
    # --------------------------------------------------------
    min_silence_frames = int(np.floor(min_silence_duration_ms / frame_duration_ms))

    merged_segments = []
    cur_start, cur_end = segments_frames[0]

    for next_start, next_end in segments_frames[1:]:
        gap = next_start - cur_end
        if gap <= min_silence_frames:
            # Merge with current segment
            cur_end = next_end
        else:
            merged_segments.append((cur_start, cur_end))
            cur_start, cur_end = next_start, next_end

    merged_segments.append((cur_start, cur_end))

    # --------------------------------------------------------
    # 8. Convert frame indices → sample indices in ORIGINAL sample_rate
    # --------------------------------------------------------
    sr_ratio = float(sample_rate) / float(target_sr)
    speech_timestamps = []

    for start_f, end_f in merged_segments:
        start_sample_48k = start_f * frame_size
        end_sample_48k = end_f * frame_size

        # Map to original sample_rate used by downstream code
        start_sample_orig = int(start_sample_48k * sr_ratio)
        end_sample_orig = int(end_sample_48k * sr_ratio)

        speech_timestamps.append({
            "start": start_sample_orig,
            "end": end_sample_orig,
        })

    return speech_timestamps

# ------------------------------------------------------------
# Extract segments based on timestamps
# ------------------------------------------------------------

def extract_speech_segments(waveform, speech_timestamps):
    """
    Given Silero timestamps, slice the waveform into speech-only chunks.

    Returns:
        List of waveform tensors, each segment is a speech region.
    """

    segments = []
    for ts in speech_timestamps:
        start = ts["start"]
        end = ts["end"]
        segments.append(waveform[:, start:end])

    return segments


def extract_nonspeech_segments(waveform, speech_timestamps):
    """
    Given Silero timestamps (speech regions), compute the 
    complementary non-speech regions.

    Returns:
        List of waveform tensors, each segment is a non-speech region.
    """

    nonspeech_segments = []
    length = waveform.shape[1]

    # If no speech: entire clip is non-speech
    if len(speech_timestamps) == 0:
        nonspeech_segments.append(waveform)
        return nonspeech_segments

    # Track previous end
    prev_end = 0

    for ts in speech_timestamps:
        start = ts["start"]
        if prev_end < start:
            nonspeech_segments.append(waveform[:, prev_end:start])
        prev_end = ts["end"]

    # Check for segment after last speech timestamp
    if prev_end < length:
        nonspeech_segments.append(waveform[:, prev_end:length])

    return nonspeech_segments

def stitch_segments(segments_list):
    
    if len(segments_list) == 0:
        return None

    # If there is only one segment, return it unchanged
    if len(segments_list) == 1:
        return segments_list[0]

    # Concatenate along the time dimension (dim=1)
    stitched = torch.cat(segments_list, dim=1)

    return stitched