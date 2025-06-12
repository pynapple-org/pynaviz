import pathlib

import nox


@nox.session(name="linters")
def linters(session):
    """Run linters"""
    session.run("ruff", "check", "src", "--ignore", "D")


@nox.session(name="tests")
def tests(session):
    """Run the test suite."""
    video_dir = pathlib.Path(__file__).parent.resolve() / "tests/test_video"
    video_dir.mkdir(exist_ok=True)
    generated_video = [f"numbered_video{ext}" for ext in [".mp4", ".avi", ".mkv"]]
    is_in_dir = all(name in  [*video_dir.iterdir()] for name in generated_video)
    session.log(f"video found {is_in_dir}")
    if not is_in_dir:
        session.log("Generating numbered videos...")
        session.run(
            "python",
            f"{video_dir.parent / 'generate_numbered_videos.py'}"
        )
    session.run(
        "pytest",
        env={"WGPU_FORCE_OFFSCREEN": "1"},
    )
