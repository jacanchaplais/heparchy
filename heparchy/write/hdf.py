"""
`heparchy.write.hdf`
===================

Provides the interface to write HEP data to the heparchy HDF5 format.
"""
import functools as fn
import typing as ty
import warnings
from enum import Enum
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
from h5py import File, Group

import heparchy as hrc
from heparchy.utils import chunk_key_format, event_key_format

from . import base

__all__ = [
    "Compression",
    "WriteOnlyError",
    "OverwriteWarning",
    "MapWriter",
    "HdfEventWriter",
    "HdfProcessWriter",
    "HdfWriter",
]

_WRITE_ONLY_MSG = "Attribute is write-only."

MapGeneric = ty.TypeVar("MapGeneric")
IterItem = ty.TypeVar("IterItem")
WriterType = ty.Union["HdfEventWriter", "HdfProcessWriter"]
ExportType = ty.TypeVar("ExportType", ty.Callable, ty.Type)


class Compression(Enum):
    """Sets the compression algorithm used to store the datasets.
    :group: hepwrite
    """

    LZF = "lzf"
    GZIP = "gzip"


class WriteOnlyError(RuntimeError):
    """Raised when trying to access write-only data.
    :group: hepwrite
    """


class OverwriteWarning(RuntimeWarning):
    """A warning to be raised when a user writes a piece of data twice.
    :group: hepwrite
    """


def _mk_dset(
    grp: Group,
    name: str,
    data: base.AnyVector,
    shape: ty.Tuple[int, ...],
    dtype: npt.DTypeLike,
    compression: Compression,
    compression_level: ty.Optional[int],
) -> None:
    """Generic dataset creation and population function.
    Wrap in methods exposed to the user interface.
    """
    if name in grp:
        warnings.warn(f"Overwriting {name}", OverwriteWarning)
        del grp[name]
    # check data can be broadcasted to dataset:
    if data.squeeze().shape != shape:
        raise ValueError(
            f"Input data shape {data.shape} "
            f"incompatible with dataset shape {shape}."
        )
    kwargs: dict[str, ty.Any] = dict(
        name=name,
        shape=shape,
        dtype=dtype,
        shuffle=True,
        compression=compression.value,
    )
    cmprs_lvl = compression_level
    if cmprs_lvl is not None:
        kwargs["compression_opts"] = cmprs_lvl
    dset = grp.create_dataset(**kwargs)
    dset[...] = data


def _mask_setter(writer: WriterType, name: str, data: base.BoolVector) -> None:
    if not isinstance(writer, HdfEventWriter):
        raise ValueError("Can't set masks on processes")
    writer._set_num_pcls(data)
    _mk_dset(
        writer._mask_grp,
        name=name,
        data=data,
        shape=data.shape,
        dtype="<?",
        compression=writer._proc._file_obj._cmprs,
        compression_level=writer._proc._file_obj._cmprs_lvl,
    )


def _custom_setter(writer: WriterType, name: str, data: base.AnyVector) -> None:
    if not isinstance(writer, HdfEventWriter):
        raise ValueError("Can't set custom datasets on processes")
    _mk_dset(
        writer._custom_grp,
        name=name,
        data=data,
        shape=data.shape,
        dtype=data.dtype,
        compression=writer._proc._file_obj._cmprs,
        compression_level=writer._proc._file_obj._cmprs_lvl,
    )


def _meta_setter(writer: WriterType, name: str, data: ty.Any) -> None:
    writer._grp.attrs[name] = data


class MapWriter(ty.Generic[MapGeneric], ty.MutableMapping[str, MapGeneric]):
    """Provides a dictionary-like interface for writing user-named
    datasets or metadata.

    :group: hepwrite

    Parameters
    ----------
    writer : HdfProcessWriter or HdfEventWriter
        The writer instance to which the MapWriter will be attached.
    setter_func : callable
        The strategy used to write the user-named data with the writer.

    Raises
    ------
    WriteOnlyError
        Dictionary-like behaviour which requires read-access for
        datasets, as access to the data is strictly write-only.

    Notes
    -----
    This is a concrete subclass of collections.abc.MutableMapping, and
    thus includes all functionality expected of a dictionary, including
    iteration over keys, subscriptable setters and deleters of datasets,
    etc.
    """

    def __init__(
        self,
        writer: WriterType,
        setter_func: ty.Callable[[WriterType, str, MapGeneric], None],
    ) -> None:
        self.writer = writer
        self._names: ty.Set[str] = set()
        self._setter_func = setter_func

    def __repr__(self) -> str:
        dset_repr = "<Write-Only Data>"
        kv = ", ".join(map(lambda k: f"'{k}': {dset_repr}", self._names))
        return "{" + f"{kv}" + "}"

    def __len__(self) -> int:
        return len(self._names)

    def __getitem__(self, name: str) -> MapGeneric:
        raise WriteOnlyError("Value is write-only.")

    def __setitem__(self, name: str, data: MapGeneric) -> None:
        self._setter_func(self.writer, name, data)
        self._names.add(name)

    def __delitem__(self, name: str) -> None:
        self._names.remove(name)
        if self._setter_func.__name__ == "_meta_setter":
            del self.writer._grp.attrs[name]
            return
        del self.writer._grp[name]

    def __iter__(self) -> ty.Iterator[str]:
        yield from self._names

    def _flush(self) -> ty.Tuple[str, ...]:
        data = tuple(self._names)
        self._names = set()
        return data


class HdfEventWriter:
    """Context manager interface to create and write events.

    :group: hepwrite

    Attributes
    ----------
    masks : MapWriter[numpy.ndarray[bool]]
        Write-only dictionary-like interface to set boolean masks
        over the particle datasets.
    custom : MapWriter[numpy.ndarray]
        Write-only dictionary-like interface to set user-defined
        custom datasets.
    custom_meta : MapWriter
        Write-only dictionary-like interface to set user-defined
        metadata on the event.

    Examples
    --------
    Store custom metadata to an event:

        >>> import heparchy
        >>>
        >>> with heparchy.write.HdfWriter("showers.hdf5") as hep_file:
        >>>     with hep_file.new_process("higgs") as proc:
        >>>         with proc.new_event() as event:
        >>>             event.custom_meta["num_jets"] = 4

    """

    def __init__(self, proc: "HdfProcessWriter") -> None:
        self._proc = proc  # pointer to parent group obj
        self._idx = proc._evt_idx  # index for current event
        self._num_pcls: ty.Optional[int] = None
        self._num_edges = 0
        self._grp: Group
        self._custom_grp: Group
        self._mask_grp: Group
        self.masks: MapWriter[base.BoolVector]
        self.custom: MapWriter[base.AnyVector]
        self.custom_meta: MapWriter[ty.Any]

    def __enter__(self: "HdfEventWriter") -> "HdfEventWriter":
        self._grp = self._proc._grp.create_group(
            event_key_format(self._idx, self._proc._file_obj._evts_per_chunk)
        )
        self._custom_grp = self._grp.create_group("custom")
        self._mask_grp = self._grp.create_group("masks")
        self.masks = MapWriter(self, _mask_setter)
        self.custom = MapWriter(self, _custom_setter)
        self.custom_meta = MapWriter(self, _meta_setter)
        self._mk_dset = fn.partial(
            _mk_dset,
            grp=self._grp,
            compression=self._proc._file_obj._cmprs,
            compression_level=self._proc._file_obj._cmprs_lvl,
        )
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        # increasing the index for the group to create next event
        # also acts as running counter for number events
        if self._num_pcls is None:
            self._num_pcls = 0
        self._grp.attrs["num_pcls"] = self._num_pcls
        self._grp.attrs["mask_keys"] = self.masks._flush()
        self._grp.attrs["custom_keys"] = self.custom._flush()
        self._grp.attrs["custom_meta_keys"] = self.custom_meta._flush()
        self._proc._evt_idx += 1

    def _set_num_pcls(self, data: base.AnyVector) -> None:
        shape = data.shape
        num_pcls = shape[0]

        if self._num_pcls is None:
            self._num_pcls = num_pcls
        elif num_pcls != self._num_pcls:
            raise ValueError(
                "Datasets within same event must have the same "
                "number of particles (rows, if 2D array). "
                f"Previous datasets had {self._num_pcls}, "
                f"attempted dataset would have {num_pcls}."
            )
        else:
            return

    @property
    def edges(self) -> ty.NoReturn:
        """Structured array of COO edge representation of graph, with
        fields 'src' and 'dst'.
        """
        raise WriteOnlyError(_WRITE_ONLY_MSG)

    @edges.setter
    def edges(self, data: base.VoidVector) -> None:
        dtype = np.dtype([("src", "<i4"), ("dst", "<i4")])
        if not data.dtype.names:
            data = data.view(dtype).reshape(-1)
        self._mk_dset(name="edges", data=data, shape=data.shape, dtype=dtype)
        self._num_edges = len(data)

    @property
    def edge_weights(self) -> ty.NoReturn:
        """Weights attributed to each edge."""
        raise WriteOnlyError(_WRITE_ONLY_MSG)

    @edge_weights.setter
    def edge_weights(self, data: base.DoubleVector) -> None:
        num_edges = self._num_edges
        num_weights = len(data)
        if (num_edges == 0) or (self._num_edges != num_weights):
            raise ValueError(
                f"Incompatible shapes. Number of edges = {num_edges} and "
                + f"number of weights = {num_weights}. Must be same size."
            )
        self._mk_dset(
            name="edge_weights",
            data=data,
            shape=data.shape,
            dtype="<f8",
        )

    @property
    def pmu(self) -> ty.NoReturn:
        """Structured array providing particle 4-momenta, with fields
        'x', 'y', 'z', 'e'.
        """
        raise WriteOnlyError(_WRITE_ONLY_MSG)

    @pmu.setter
    def pmu(self, data: base.VoidVector) -> None:
        dtype = np.dtype([(name, "<f8") for name in "xyze"])
        if not data.dtype.names:
            data = data.view(dtype).reshape(-1)
        self._set_num_pcls(data)
        self._mk_dset(name="pmu", data=data, shape=data.shape, dtype=dtype)

    @property
    def color(self) -> ty.NoReturn:
        """A structured array with fields 'color' and 'anti-color'."""
        raise WriteOnlyError(_WRITE_ONLY_MSG)

    @color.setter
    def color(self, data: base.AnyVector) -> None:
        dtype = np.dtype([("color", "<i4"), ("anticolor", "<i4")])
        if not data.dtype.names:
            data = data.view(dtype).reshape(-1)
        self._set_num_pcls(data)
        self._mk_dset(name="color", data=data, shape=data.shape, dtype=dtype)

    @property
    def pdg(self) -> ty.NoReturn:
        """PDG codes identifying each particle in the event."""
        raise WriteOnlyError(_WRITE_ONLY_MSG)

    @pdg.setter
    def pdg(self, data: base.IntVector) -> None:
        self._set_num_pcls(data)
        self._mk_dset(
            name="pdg",
            data=data,
            shape=(self._num_pcls,),
            dtype="<i4",
        )

    @property
    def status(self) -> ty.NoReturn:
        """Status codes, describing the role of each particle in the
        event.
        """
        raise WriteOnlyError(_WRITE_ONLY_MSG)

    @status.setter
    def status(self, data: base.HalfIntVector) -> None:
        self._set_num_pcls(data)
        self._mk_dset(
            name="status",
            data=data,
            shape=(self._num_pcls,),
            dtype="<i2",
        )

    @property
    def helicity(self) -> ty.NoReturn:
        """Spin polarisations of every particle in the event."""
        raise WriteOnlyError(_WRITE_ONLY_MSG)

    @helicity.setter
    def helicity(self, data: base.HalfIntVector) -> None:
        self._set_num_pcls(data)
        self._mk_dset(
            name="helicity",
            data=data,
            shape=(self._num_pcls,),
            dtype="<i2",
        )


class HdfProcessWriter:
    """Context manager interface to create and write processes.

    :group: hepwrite

    Attributes (write-only)
    -----------------------
    custom_meta : MapWriter
        Write-only dictionary-like interface to set custom metadata to
        the process.

    Examples
    --------
    Wrap iterator with event_iter to obtain HdfEventWriter object, which
    stores pdg and pmu data to each event.

        >>> import heparchy
        >>> from showerpipe.generators import PythiaGenerator
        >>>
        >>> gen = PythiaGenerator("pythia-card.cmnd", "higgs.lhe.gz")
        >>> with heparchy.write.HdfWriter("showers.hdf5") as hep_file:
        >>>     with hep_file.new_process("higgs") as proc:
        >>>         for event_out, event_in in proc.event_iter(gen):
        >>>             # event_in is generated from Pythia
        >>>             event_out.pdg = event_in.pdg
        >>>             event_out.pmu = event_in.pmu

    """

    def __init__(self, file_obj: "HdfWriter", key: str) -> None:
        self._file_obj = file_obj
        self.key = key
        self._evt_idx = 0
        self._parent: Group
        self._grp: Group
        self.custom_meta: MapWriter[ty.Any]

    def _evtgrp_iter(self):
        chunk = 0
        while True:
            grp = self._parent.create_group(chunk_key_format(chunk))
            for _ in range(self._file_obj._evts_per_chunk):
                yield grp
            chunk = chunk + 1

    def __enter__(self: "HdfProcessWriter") -> "HdfProcessWriter":
        self._parent = self._file_obj._buffer.create_group(self.key)
        self._events = self._evtgrp_iter()
        self.custom_meta = MapWriter(self, _meta_setter)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        # count all of the events and write to attribute
        self._grp.attrs["custom_meta_keys"] = self.custom_meta._flush()
        self._parent.attrs["num_evts"] = self._evt_idx

    @property
    def process_string(self) -> ty.NoReturn:
        """String representing the hard process (eg. MadGraph format)."""
        raise WriteOnlyError(_WRITE_ONLY_MSG)

    @process_string.setter
    def process_string(self, value: str) -> None:
        self._grp.attrs["process"] = value

    @property
    def signal_pdgs(self) -> ty.NoReturn:
        """PDG codes within the hard process considered as 'signal'."""
        raise WriteOnlyError(_WRITE_ONLY_MSG)

    @signal_pdgs.setter
    def signal_pdgs(self, value: ty.Sequence[int]) -> None:
        self._grp.attrs["signal_pdgs"] = value

    @property
    def com_energy(self) -> ty.NoReturn:
        """A tuple whose first element is the centre-of-mass collision
        energy for the hard process, and the second element is the unit
        that this energy is given in.
        """
        raise WriteOnlyError(_WRITE_ONLY_MSG)

    @com_energy.setter
    def com_energy(self, value: ty.Tuple[float, str]) -> None:
        self._grp.attrs["com_e"] = value[0]
        self._grp.attrs["e_unit"] = value[1]

    def new_event(self) -> HdfEventWriter:
        self._grp = next(self._events)
        return HdfEventWriter(self)

    def event_iter(
        self, iterable: ty.Iterable[IterItem]
    ) -> ty.Iterator[ty.Tuple[HdfEventWriter, IterItem]]:
        """Wraps an iteratable object, returning a new iterator which
        yields a new writeable event object followed by the value
        obtained from the passed iterator.

        :group: hepwrite

        Parameters
        ----------
        iterable : Iterable[IterItem]
            Any iterable object, eg. tuples, lists, generators, etc.

        Returns
        -------
        event : HdfEventWriter
            Writeable event object.
        value : IterItem
            The value yielded by the input iterable.
        """
        for value in iterable:
            with self.new_event() as event:
                yield event, value


class HdfWriter:
    """Create a new heparchy hdf5 file object with write access.

    :group: hepwrite

    Parameters
    ----------
    path : Path or str
        Filepath for output.
    compression : Compression or str
        Supports "gzip" or "lzf" compression for datasets. Default is
        gzip.
    compression_level : int, optional
        Integer between 1 - 9 setting gzip compression factor. Ignored
        if compression type is lzf. Default is 4.
    evts_per_chunk : int
        Number of events which should be grouped into a chunk. Grouping
        the events vastly improves read speed for processes which store
        large numbers of events, _eg._ 1e+5 or more. Default is 1000.

    Examples
    --------
    Open a new heparchy HDF5 file, and create a process within it:

        >>> import heparchy
        >>>
        >>> with heparchy.write.HdfWriter("showers.hdf5") as hep_file:
        >>>     with hep_file.new_process("higgs") as proc:
        >>>         ...
    """

    def __init__(
        self,
        path: ty.Union[Path, str],
        compression: ty.Union[str, Compression] = Compression.GZIP,
        compression_level: ty.Optional[int] = 4,
        evts_per_chunk: int = 1000,
    ) -> None:
        self.path = Path(path)
        self._buffer: File
        if isinstance(compression, str):
            compression = Compression(compression.lower())
        self._cmprs = compression
        if compression is Compression.LZF:
            compression_level = None
        self._cmprs_lvl = compression_level
        self._evts_per_chunk = evts_per_chunk

    def __enter__(self: "HdfWriter") -> "HdfWriter":
        self._buffer = h5py.File(self.path, "w", libver="latest")
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self._buffer.attrs["layout"] = "heparchy"
        self._buffer.attrs["version_tuple"] = tuple(map(str, hrc.__version_tuple__))
        self._buffer.attrs["version"] = hrc.__version__
        self._buffer.attrs["evts_per_chunk"] = self._evts_per_chunk
        self._buffer.close()

    def new_process(self, name: str) -> HdfProcessWriter:
        """Returns a context handler object for storage of data in
        a given hard process.

        Events can be iteratively added to this process by repeatedly
        calling the ``new_event()`` method, which itself returns a
        context handler object.

        :group: hepwrite

        Parameters
        ----------
        key : str
            Arbitrary name, used to look up data with reader.

        Returns
        -------
        process : HdfProcessWriter
            Context manager for creating processes within the file,
            providing methods for adding metadata and events.
            See heparchy.write.HdfProcessWriter for more.
        """
        return HdfProcessWriter(self, key=name)
