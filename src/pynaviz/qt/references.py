"""
Store references to NWB files and EphysReaders.

This module defines dataclasses to hold references to NWB files and EphysReaders, along with the specific variable keys being accessed. These references are used
in the Qt widgets to keep the NWB file and EphysReader objects alive, preventing their associated HDF5 datasets and lazy-loaded data from being garbage collected while the widgets are still in use.

"""


from dataclasses import dataclass

import pynapple as nap


@dataclass
class NWBReference:
    """Store reference to NWB file object and variable key.

    Keeps the NWB file (and its IO handle) alive to prevent HDF5 datasets
    from being closed during garbage collection.
    """
    nwb_file: nap.io.NWBFile
    key: str


@dataclass
class EphysReference:
    """Store reference to EphysReader and variable key.

    Keeps the EphysReader (and its Neo reader/blocks) alive to prevent
    lazy-loaded data from being garbage collected.
    """
    ephys_reader: nap.EphysReader
    key: str
