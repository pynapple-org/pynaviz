import pathlib


def download_from_osf(file_id: str, nwb_path: str | pathlib.Path):
    """
    Download an NWB file from OSF (Open Science Framework).

    Downloads a file from OSF using its file ID and saves it to the specified path.
    If the file already exists at the target path, the download is skipped.

    Parameters
    ----------
    file_id : str
        The OSF file identifier used to construct the download URL.
    nwb_path : str or pathlib.Path
        The local path where the NWB file should be saved.

    Notes
    -----
    The function downloads the file in chunks of 1 MB to handle large files
    efficiently without loading the entire file into memory.
    """
    # runtime import (this is needed for running the linters without
    # installing the test environment).
    import requests
    nwb_path = pathlib.Path(nwb_path)

    if nwb_path.exists():
        print("Using existing nwb file.")
        return

    r = requests.get(f"https://osf.io/download/{file_id}", stream=True)
    block_size = 1024 * 1024
    with open(nwb_path, 'wb') as f:
        for data in r.iter_content(block_size):
            f.write(data)
    return

if __name__ == "__main__":
    import sys
    file_id, nwb_path =  sys.argv[1:3]
    download_from_osf(file_id, nwb_path)
