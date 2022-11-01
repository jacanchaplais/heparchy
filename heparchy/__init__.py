"""
`heparchy`
==========

The hierarchically formatted high energy physics IO library.
"""
import warnings

from heparchy import read, write
from heparchy._version import __version__, __version_tuple__


__all__ = ["read", "write", "__version__", "__version_tuple__"]
warnings.filterwarnings("once", category=DeprecationWarning)
