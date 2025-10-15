import pathlib

import av
import numpy as np

EXTENSION_TO_CODEC = {
    ".wav": "pcm_s16le",    # raw PCM
    ".mp3": "libmp3lame",   # MP3 (if available in your FFmpeg build)
    ".flac": "flac",        # FLAC
}

def generate_noise_audio(
    output_path: str | pathlib.Path = "test_audio/noise_audio.wav",
    samplerate: int = 44100,
    duration_sec: float = 2.0,
    amplitude: float = 0.5,   # peak scaling after normalization
):
    output_path = pathlib.Path(__file__).resolve().parent / output_path
    output_path.parent.mkdir(exist_ok=True)

    codec_name = EXTENSION_TO_CODEC.get(output_path.suffix.lower())
    if codec_name is None:
        raise ValueError(f"Unsupported audio format: {output_path.suffix}")

    num_samples = int(samplerate * duration_sec)

    # Deterministic Gaussian noise (mono): shape (channels, samples)
    rng = np.random.default_rng(seed=12345)
    print("samples", num_samples)
    waveform = rng.normal(size=(1, num_samples))
    # Normalize to [-1, 1] peak, then scale
    waveform /= np.max(np.abs(waveform))
    waveform *= amplitude

    # Convert to int16 for s16 encoding
    waveform_int16 = (waveform * 32767).astype(np.int16)
    with av.open(output_path, mode="w") as container:
        stream = container.add_stream(codec_name, rate=samplerate)

        # Explicitly set both the encoder's expected format/layout (codec_context)
        # and the actual format/layout of the frame we feed in.
        # This avoids implicit conversions by FFmpeg, which can cause extra copies
        # or subtle mismatches — especially with compressed formats like MP3/FLAC
        # that have strict frame size requirements.

        stream.codec_context.sample_rate = samplerate
        stream.codec_context.format = "s16"
        stream.codec_context.layout = "mono"

        frame = av.AudioFrame.from_ndarray(waveform_int16, format="s16", layout="mono")
        frame.sample_rate = samplerate

        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):  # flush encoder
            container.mux(packet)
        container.close()

    print(f"Saved audio to {output_path}")

if __name__ == "__main__":
    generate_noise_audio("test_audio/noise_audio.wav")
    generate_noise_audio("test_audio/noise_audio.mp3")
    generate_noise_audio("test_audio/noise_audio.flac")
