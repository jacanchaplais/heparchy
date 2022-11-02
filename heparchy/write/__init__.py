"""
`heparchy.write`
================

Interfaces to write to standard particle physics formats,
and the heparchy formats.

Notes
-----
Currently only implemented for hdf5.
"""
from . import hdf
from .hdf import *


__all__ = hdf.__all__.copy()
