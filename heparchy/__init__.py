"""
`heparchy`
==========

The hierarchically formatted high energy physics IO library.
"""
import contextlib as ctx
import typing as ty
import warnings
from pathlib import Path

from heparchy import read, write
from heparchy._version import __version__, __version_tuple__

__all__ = ["read", "write", "open_file", "__version__", "__version_tuple__"]
warnings.filterwarnings("once", category=DeprecationWarning)


@ty.overload
def open_file(
    path: ty.Union[str, Path], mode: ty.Literal["r"]
) -> ty.ContextManager[read.hdf.HdfReader]:
    ...


@ty.overload
def open_file(
    path: ty.Union[str, Path], mode: ty.Literal["r"], process: None
) -> ty.ContextManager[read.hdf.HdfReader]:
    ...


@ty.overload
def open_file(
    path: ty.Union[str, Path], mode: ty.Literal["r"], process: str
) -> ty.ContextManager[read.hdf.HdfProcessReader]:
    ...


@ty.overload
def open_file(
    path: ty.Union[str, Path], mode: ty.Literal["w"]
) -> ty.ContextManager[write.hdf.HdfWriter]:
    ...


@ty.overload
def open_file(
    path: ty.Union[str, Path], mode: ty.Literal["w"], process: None
) -> ty.ContextManager[write.hdf.HdfWriter]:
    ...


@ty.overload
def open_file(
    path: ty.Union[str, Path], mode: ty.Literal["w"], process: str
) -> ty.ContextManager[write.hdf.HdfProcessWriter]:
    ...


@ctx.contextmanager
def open_file(
    path: ty.Union[str, Path],
    mode: ty.Literal["r", "w"],
    process: ty.Optional[str] = None,
):
    """High level file manager for reading and writing HEP data to HDF5.

    Parameters
    ----------
    path : Path or str
        Path on disk where the file is to be opened.
    mode : {'r', 'w'}
        Whether to open the file in 'r' (read) or 'w' (write) mode.
    process : str, optional
        If provided, will provide the handler to the process indexed
        with the value.

    Yields
    ------
    HdfReader or HdfProcessReader or HdfWriter or HdfProcessWriter
        Heparchy's file handler for accessing the HDF5 file.
    """
    if mode not in {"r", "w"}:
        raise ValueError(f'Mode {mode} not known. Please use "r" or "w".')
    stack = ctx.ExitStack()
    if mode == "r":
        f = stack.enter_context(read.hdf.HdfReader(path))
        if process is not None:
            f = f[process]
    elif mode == "w":
        f = stack.enter_context(write.hdf.HdfWriter(path))
        if process is not None:
            f = stack.enter_context(f.new_process(process))
    try:
        yield f
    finally:
        stack.close()
