"""
`heparchy.read`
===============

Interfaces to read from standard particle physics formats,
and the heparchy formats.

Notes
-----
Currently only implemented for hepmc and hdf5.
"""
from . import hdf
from .hdf import *
import heparchy.read.hepmc as hepmc


HepMC = hepmc.HepMC


class Wiggle:
    pass


__all__ = ["HepMC", "Wiggle"]
__all__.extend(hdf.__all__)
