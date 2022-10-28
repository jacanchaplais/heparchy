"""
`heparchy.write`
================

Interfaces to write to standard particle physics formats,
and the heparchy formats.

Notes
-----
Currently only implemented for hdf5.
"""
import heparchy.write.hdf as hdf


HdfWriter = hdf.HdfWriter

__all__ = ["HdfWriter"]
