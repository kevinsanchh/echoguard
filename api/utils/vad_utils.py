import torch
import torchaudio

# ------------------------------------------------------------
# Silero VAD Model Loader
# ------------------------------------------------------------

def load_vad_model():
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad',force_reload=False)

    (get_speech_ts,
     _,              # vad_iterator (unused)
     read_audio,
     *_) = utils     # unpack other unused utils

    vad_helpers = {
        "get_speech_ts": get_speech_ts,
        "read_audio": read_audio
    }

    return model, vad_helpers


# ------------------------------------------------------------
# Run VAD on a waveform (PyTorch tensor)
# ------------------------------------------------------------

def run_vad_on_waveform(waveform, model, vad_helpers, sample_rate=16000):
    """
    Applies Silero VAD to a loaded waveform.

    Parameters:
        waveform (torch.Tensor): Tensor shape [1, num_samples]
        model: Silero VAD model
        vad_helpers: dict of helper functions from Silero
        sample_rate: audio sample rate (default 16 kHz)

    Returns:
        speech_timestamps (list): List of dicts with speech regions
    """

    get_speech_ts = vad_helpers["get_speech_ts"]

    # Silero expects mono audio: [num_samples]
    if len(waveform.shape) == 2:
        waveform = waveform.squeeze(0)

    speech_timestamps = get_speech_ts(waveform, model, sampling_rate=sample_rate)

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