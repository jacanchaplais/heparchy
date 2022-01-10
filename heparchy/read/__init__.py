"""Interfaces to read from standard particle physics formats,
and the heparchy formats.

Notes
-----
Currently only implemented for hepmc and hdf5.
"""

from .hdf import HdfReader
from .hepmc import HepMC
