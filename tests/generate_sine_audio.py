import numpy as np
import av
import pathlib

EXTENSION_TO_CODEC = {
    ".wav": "pcm_s16le",    # raw PCM
    ".mp3": "libmp3lame",   # MP3
    ".flac": "flac",        # FLAC
}


def generate_sine_audio(
    output_path: str | pathlib.Path = "test_audio/sine_audio.wav",
    samplerate: int = 44100,
    duration_sec: float = 2.0,
    frequency: float = 440.0,
    amplitude: float = 0.5,
):
    output_path = pathlib.Path(__file__).resolve().parent / output_path
    output_path.parent.mkdir(exist_ok=True)

    codec_name = EXTENSION_TO_CODEC.get(output_path.suffix.lower())
    if codec_name is None:
        raise ValueError(f"Unsupported audio format: {output_path.suffix}")

    num_samples = int(samplerate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    waveform = amplitude * np.sin(2 * np.pi * frequency * t)
    waveform_int16 = (waveform * 32767).astype(np.int16)

    # Reshape to (channels, samples) as required by AudioFrame
    mono_waveform = np.expand_dims(waveform_int16, axis=0)

    container = av.open(output_path, mode="w")
    stream = container.add_stream(codec_name, rate=samplerate)
    stream.codec_context.sample_rate = samplerate
    stream.codec_context.format = "s16"
    stream.codec_context.layout = "mono"

    frame = av.AudioFrame.from_ndarray(mono_waveform, format="s16", layout="mono")
    frame.sample_rate = samplerate

    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):  # flush encoder
        container.mux(packet)

    container.close()
    print(f"Saved audio to {output_path}")


if __name__ == "__main__":
    generate_sine_audio("test_audio/sine_audio.wav")
    generate_sine_audio("test_audio/sine_audio.mp3")
    generate_sine_audio("test_audio/sine_audio.flac")
