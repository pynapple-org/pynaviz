import argparse
from pathlib import Path

import pynaviz
from pynaviz import scope


def json_file(path_str):
    path = Path(path_str)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    if path.suffix.lower() != ".json":
        raise argparse.ArgumentTypeError(f"Layout file must be a .json file: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        prog="pynaviz",
        description=(
            "Visualize and synchronize time series using pynapple. "
            "Accepted inputs: .nwb, .npz, movie files (.mp4/.avi/.mov/.mkv), "
            "electrophysiology files (.plx, .ncs, .smr, …), directories containing "
            "electrophysiology recordings, and .json layout files. "
            "If no files are provided, an empty viewer is launched."
        ),
        epilog="Example usage: pynaviz data1.nwb movie.mp4 -l layout.json"
    )

    parser.add_argument(
        "files",
        type=Path,
        nargs="*",
        help="Paths to .nwb, .npz, movie, ephys files, or directories with ephys recordings"
    )

    parser.add_argument(
        "-l", "--layout",
        type=json_file,
        help="Path to a JSON layout file"
    )

    parser.add_argument(
        "-f", "--format",
        type=str,
        default=None,
        metavar="FORMAT",
        help=(
            "Neo IO format for EphysReader inputs (e.g. 'PlexonIO', 'NeuroScopeIO'). "
            "Applied to all ephys files/directories. When omitted, format is auto-detected."
        )
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {getattr(pynaviz, '__version__', 'dev')}"
    )

    args = parser.parse_args()

    files = args.files or []
    missing = [f for f in files if not f.exists()]
    if missing:
        parser.error(f"Path(s) not found: {', '.join(str(f) for f in missing)}")

    scope(
        [str(f) for f in files],
        layout_path=str(args.layout) if args.layout else None,
        ephys_format=args.format,
    )

if __name__ == "__main__":
    main()
