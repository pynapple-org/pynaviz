import pathlib

import av
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

EXTENSION_TO_CODEC = {
    ".mp4": "mpeg4",
    ".avi": "mpeg4",
    ".mkv": "libx264",
    ".webm": "vp9",
    ".ogv": "libtheora",
    ".mov": "libx264",
}


def generate_numbered_video(output_path: str | pathlib.Path="test_video/numbered_video.mp4", num_frames=100, fps=30, width=640, height=480):
    output_path = pathlib.Path(__file__).resolve().parent / output_path
    output_path.parent.mkdir(exist_ok=True)
    # Set up container and stream
    with av.open(output_path, mode='w') as container:
        codec = EXTENSION_TO_CODEC[output_path.suffix]
        stream = container.add_stream(codec, rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'

        # Prepare figure
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        canvas = FigureCanvas(fig)
        ax.axis("off")

        for i in range(num_frames):
            ax.clear()
            ax.axis("off")
            ax.text(0.5, 0.5, str(i), fontsize=60, ha='center', va='center', transform=ax.transAxes)

            canvas.draw()
            buf = np.asarray(canvas.buffer_rgba())[:, :, :3]  # RGB image

            # Convert to frame
            frame = av.VideoFrame.from_ndarray(buf, format='rgb24')
            frame = frame.reformat(width, height, format='yuv420p')

            # Encode and mux
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    print(f"Saved video to {output_path}")

if __name__ == "__main__":
    generate_numbered_video("test_video/numbered_video.mp4")
    generate_numbered_video("test_video/numbered_video.avi")
    generate_numbered_video("test_video/numbered_video.mkv")
