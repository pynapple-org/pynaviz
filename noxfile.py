import pathlib

import nox


@nox.session(name="linters", reuse_venv=True)
def linters(session):
    """Run linters"""
    session.install("-e", ".[dev]")
    session.run("ruff", "check", "src", "--ignore", "D")
    session.run("ruff", "check", "tests", "--ignore", "D")

@nox.session(name="linters-fix", reuse_venv=True)
def linters_fix(session):
    """Run linters and auto-fix issues"""
    session.install("-e", ".[dev]")
    session.run("ruff", "check", "src", "--ignore", "D", "--fix")
    session.run("ruff", "check", "tests", "--ignore", "D", "--fix")

@nox.session(name="tests", reuse_venv=True)
def tests(session):
    """Run the test suite."""
    # session.log("install")
    session.install("-e", ".[dev]")
    tests_path = pathlib.Path(__file__).parent.resolve() / "tests"

    # generate sample videos
    video_dir = tests_path / "test_video"
    video_dir.mkdir(exist_ok=True)
    generated_video = [f"numbered_video{ext}" for ext in [".mp4", ".avi", ".mkv"]]
    is_in_dir = all((video_dir / name).exists() for name in generated_video)
    session.log(f"video found {is_in_dir}")
    if not is_in_dir:
        session.log("Generating numbered videos...")
        session.run(
            "python",
            f"{video_dir.parent / 'generate_numbered_video.py'}"
        )
    # generate sample audios
    audio_dir = tests_path / "test_audio"
    audio_dir.mkdir(exist_ok=True)
    generated_audio = [f"noise_audio{ext}" for ext in [".mp3", ".wav", ".flac"]]
    is_in_dir = all((audio_dir / name).exists() for name in generated_audio)
    if not is_in_dir:
        session.log("Generating noise wave audio...")
        session.run(
            "python",
            f"{audio_dir.parent / 'generate_noise_audio.py'}"
        )

    # generate screenshots
    gen_screenshot_script = tests_path / "generate_screenshots.py"
    session.log("Generating screenshots...")
    session.run(
        "python",
        gen_screenshot_script.as_posix(),
        "--type", "all",
        # "--path", "tests/screenshots",
    )
    session.log("Run Tests...")
    session.run(
        "pytest",
        env={
            "WGPU_FORCE_OFFSCREEN": "1",
        },
    )
