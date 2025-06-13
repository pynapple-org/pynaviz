import pathlib

import nox


@nox.session(name="linters")
def linters(session):
    """Run linters"""
    session.run("ruff", "check", "src", "--ignore", "D")


@nox.session(name="tests")
def tests(session):
    """Run the test suite."""
    session.log("install")
    session.install(".[dev]")
    tests_path = pathlib.Path(__file__).parent.resolve() / "tests"
    video_dir = tests_path / "test_video"
    video_dir.mkdir(exist_ok=True)
    generated_video = [f"numbered_video{ext}" for ext in [".mp4", ".avi", ".mkv"]]
    is_in_dir = all(name in  [*video_dir.iterdir()] for name in generated_video)
    session.log(f"video found {is_in_dir}")
    if not is_in_dir:
        session.log("Generating numbered videos...")
        session.run(
            "python",
            f"{video_dir.parent / 'generate_numbered_video.py'}"
        )

    gen_screenshot_script = tests_path / "generate_screenshots.py"
    session.log(f"Generating screenshots...")
    session.run(
        "python",
        gen_screenshot_script.as_posix(),
        "--type", "video",
        "--path", "tests/screenshots",
    )
    session.log(f"Run Tests...")
    session.run(
        "pytest",
        env={"WGPU_FORCE_OFFSCREEN": "1"},
    )
