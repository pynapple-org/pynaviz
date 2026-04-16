"""
Module to load variables from various sources (files, NWB files, EphysReaders, etc.)
and return a dictionary of pynapple objects or video paths.

"""

import os
import pathlib
from collections import defaultdict
from typing import Any

import pynapple as nap

from .references import EphysReference, NWBReference

EPHYS_EXTENSIONS = {".plx", ".ncs", ".smr", ".nev", ".mcd", ".dat", ".lfp", ".eeg"}


def _filter_paths(path: str, ephys_format: str | None = None) -> tuple | None:
    """Filter paths and return either a valid video path or a pynapple object."""
    p = pathlib.Path(path)

    if p.is_dir():
        try:
            data = nap.EphysReader(path, format=ephys_format)
            nap_obj_dict = {key: EphysReference(ephys_reader=data, key=key) for key in data.keys()}
            return p.name, nap_obj_dict
        except Exception as e:
            print(f"Could not load directory {path} as EphysReader: {e}")
            return None, None

    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        base_name = os.path.basename(path)
        if ext in [".mp4", ".avi", ".mov", ".mkv"]:
            return base_name, path
        elif ext == ".nwb":
            data = nap.load_file(path)
            nap_obj_dict = {}
            for key in data.keys():
                nap_obj_dict[key] = NWBReference(nwb_file=data, key=key)
            return base_name, nap_obj_dict
        elif ext == ".npz":
            data = nap.load_file(path)
            if "pynapple" in data.__module__:
                return base_name, data
            else:
                return None, None
        elif ext in EPHYS_EXTENSIONS or ephys_format is not None:
            try:
                data = nap.EphysReader(path, format=ephys_format)
                nap_obj_dict = {key: EphysReference(ephys_reader=data, key=key) for key in data.keys()}
                return base_name, nap_obj_dict
            except Exception as e:
                print(f"Could not load {path} as EphysReader: {e}")
                return None, None

    return None, None


def _extract_name_value(v: Any, ephys_format: str | None = None):
    """Return (base_name, value) if valid, otherwise (None, None)."""
    if isinstance(v, (str, pathlib.Path)):
        return _filter_paths(str(v), ephys_format=ephys_format)

    if hasattr(v, "__module__"):
        if isinstance(v, nap.NWBFile):
            nap_obj_dict = {}
            for key in v.keys():
                nap_obj_dict[key] = NWBReference(nwb_file=v, key=key)
            return v.__class__.__name__, nap_obj_dict
        if isinstance(v, nap.EphysReader):
            nap_obj_dict = {key: EphysReference(ephys_reader=v, key=key) for key in v.keys()}
            return v.name, nap_obj_dict
        if "pynapple" in v.__module__:
            return v.__class__.__name__, v
        # Import VideoHandler lazily to avoid circular import
        from .widget_plot import VideoHandler
        if "pynaviz" in v.__module__ and isinstance(v, VideoHandler):
            return v.__class__.__name__, v

    return None, None


def get_pynapple_variables(
    variables: dict | list | tuple | str | None = None,
    ephys_format: str | None = None,
) -> dict:
    if variables is None:
        return {}

    if isinstance(variables, str):
        variables = [variables]

    if isinstance(variables, nap.NWBFile):
        nap_obj_dict = {key: NWBReference(nwb_file=variables, key=key) for key in variables.keys()}
        return {variables.__class__.__name__: nap_obj_dict}

    if isinstance(variables, nap.EphysReader):
        nap_obj_dict = {key: EphysReference(ephys_reader=variables, key=key) for key in variables.keys()}
        return {variables.name: nap_obj_dict}

    new_vars = {}

    if isinstance(variables, (list, tuple)):
        name_counters = defaultdict(int)  # keep track of how many times a name was used

        for v in variables:

            base_name, value = _extract_name_value(v, ephys_format=ephys_format)

            if base_name is None:
                continue

            # Ensure unique name by appending _0, _1, etc. if needed
            count = name_counters[base_name]
            unique_name = f"{base_name}_{count}" if count > 0 else base_name

            name_counters[base_name] += 1
            new_vars[unique_name] = value

    elif isinstance(variables, dict):
        for k, v in variables.items():
            base_name, value = _extract_name_value(v, ephys_format=ephys_format)
            if base_name is not None:
                new_vars[k] = value

    return new_vars
