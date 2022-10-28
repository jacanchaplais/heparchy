"""
`heparchy.read`
===============

Interfaces to read from standard particle physics formats,
and the heparchy formats.

Notes
-----
Currently only implemented for hepmc and hdf5.
"""
import heparchy.read.hdf as hdf
import heparchy.read.hepmc as hepmc


HdfReader = hdf.HdfReader
HepMC = hepmc.HepMC


__all__ = ["HdfReader", "HepMC"]
