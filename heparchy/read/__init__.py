"""
`heparchy.read`
===============

Interfaces to read from standard particle physics formats,
and the heparchy formats.

Notes
-----
Currently only implemented for hepmc and hdf5.
"""
from .hdf import HdfEventReader, HdfProcessReader, HdfReader, MapReader, ReadOnlyError

__all__ = [
    "ReadOnlyError",
    "MapReader",
    "HdfEventReader",
    "HdfProcessReader",
    "HdfReader",
]
