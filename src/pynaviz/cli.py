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
        description="Visualize and synchronize time series using pynapple. Accepted file formats: .nwb, .npz, movie files, and .json (for layout). If no files are provided, an empty viewer is launched.",
        epilog="Example usage: pynaviz data1.nwb movie.mp4 -l layout.json"
    )

    parser.add_argument(
        "files",
        type=Path,
        nargs="*",   # zero or more files
        help="Paths to .nwb, .hdf, .npz, movie, or .json files"
    )

    parser.add_argument(
        "-l", "--layout",
        type=json_file,
        help="Path to a JSON layout file"
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
        parser.error(f"File(s) not found: {', '.join(str(f) for f in missing)}")

    scope([str(f) for f in files], layout_path=str(args.layout) if args.layout else None)

if __name__ == "__main__":
    main()
