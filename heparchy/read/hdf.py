"""
`heparchy.read.hdf`
===================

Defines the data structures to open and iterate through heparchy
formatted HDF5 files.
"""
from __future__ import annotations

import typing as ty
from collections.abc import Mapping
from functools import cached_property
from os.path import basename
from pathlib import Path

import h5py
import numpy as np
from h5py import AttributeManager, Dataset, Group

from heparchy.utils import chunk_key_format, deprecated, event_key_format

from . import base

__all__ = [
    "ReadOnlyError",
    "MapReader",
    "HdfEventReader",
    "HdfProcessReader",
    "HdfReader",
]


_NOT_NUMPY_ERR = ValueError("Stored data type is corrupted.")
_NO_EVENT_ERR = AttributeError("Event reader not pointing to event.")
_BUILTIN_PROPS: ty.Set[str] = set()
_BUILTIN_METADATA = {  # TODO: work out non-hardcoded
    "HdfEventReader": {"num_pcls", "mask_keys"},
    "HdfProcessReader": {"signal_pdgs", "process", "com_e", "e_unit"},
}

MetaDictType = ty.Dict[str, ty.Union[str, int, float, bool, base.AnyVector]]
ReaderType = ty.Union["HdfEventReader", "HdfProcessReader"]
ExportType = ty.TypeVar("ExportType", ty.Callable, ty.Type)
DataType = ty.TypeVar("DataType")
PropMethod = ty.TypeVar("PropMethod", bound=ty.Callable)
MapValue = ty.TypeVar("MapValue")


class ReadOnlyError(RuntimeError):
    """Raised when trying to write to read-only data.
    :group: hepread
    """


def _reg_event_builtin(func: PropMethod) -> PropMethod:
    _BUILTIN_PROPS.add(func.__name__)
    return func


def _stored_keys(attrs: AttributeManager, key_attr_name: str) -> ty.Iterator[str]:
    key_ds = attrs[key_attr_name]
    if not isinstance(key_ds, np.ndarray):
        raise _NOT_NUMPY_ERR
    yield from tuple(key_ds)


def _grp_key_iter(reader: Group) -> ty.Iterator[str]:
    yield from reader.keys()


def _meta_iter(reader: Group) -> ty.Iterator[str]:
    key_attr_name = "custom_meta_keys"
    if key_attr_name not in reader.attrs:
        names = set(reader.attrs.keys())
        yield from names - _BUILTIN_METADATA[reader.__class__.__name__]
    else:
        yield from _stored_keys(reader.attrs, key_attr_name)


class MapReader(ty.Generic[MapValue], Mapping[str, MapValue]):
    """Read-only dictionary-like interface to user-named heparchy
    datasets.

    :group: hepread

    Parameters
    ----------
    reader : HdfProcessReader | HdfEventReader
        The reader instance to which the MapReader will be attached.
    iter_func : callable
        The strategy used to iterate through the keys for the MapReader
        interface.

    Raises
    ------
    ReadOnlyError
        Dictionary-like behaviour which requires write-access for
        datasets, as access to the data is strictly read-only.

    Notes
    -----
    This is a concrete subclass of collections.abc.MutableMapping, and
    thus includes all functionality expected of a dictionary, including
    iteration over keys and values, subscriptable access to datasets,
    etc.
    """

    def __init__(
        self,
        reader: ReaderType,
        grp_attr_name: str,
        iter_func: ty.Callable[..., ty.Iterator[str]],
    ) -> None:
        self._reader = reader
        self._grp_attr_name = grp_attr_name
        self._iter_func = iter_func

    @property
    def _grp(self) -> Group:
        return getattr(self._reader, self._grp_attr_name)

    def __repr__(self) -> str:
        dset_repr = "<Read-Only Data>"
        kv = ", ".join(map(lambda k: f"'{k}': {dset_repr}", self))
        return "{" + f"{kv}" + "}"

    def __len__(self) -> int:
        return len(tuple(iter(self)))

    def __getitem__(self, name: str) -> MapValue:
        if name not in set(self):
            raise KeyError("No data stored with this name")
        if self._iter_func.__name__ == "_meta_iter":
            return self._grp.attrs[name]  # type: ignore
        data = self._grp[name]
        if not isinstance(data, Dataset):
            raise _NOT_NUMPY_ERR
        return data[...]  # type: ignore

    def __setitem__(self, name: str, data: MapValue) -> ty.NoReturn:
        raise ReadOnlyError("Value is read-only")

    def __delitem__(self, name: str) -> ty.NoReturn:
        raise ReadOnlyError("Value is read-only")

    def __iter__(self) -> ty.Iterator[str]:
        yield from self._iter_func(self._grp)


def _type_error_str(data: ty.Any, dtype: type) -> str:
    return (
        "Type mismatch: retrieved value should be of "
        f"type {dtype}, found instance of {type(data)}"
    )


def _check_type(data: ty.Any, dtype: ty.Type[DataType]) -> DataType:
    if not isinstance(data, dtype):
        raise TypeError(_type_error_str(data, dtype))
    return data


def _format_array_dict(
    keys: ty.Sequence[str], in_dict: ty.Dict[str, ty.Any]
) -> ty.Dict[str, base.AnyVector]:
    array_dict = dict()
    for key in keys:
        val = in_dict[key][...]
        if not isinstance(val, np.ndarray):
            raise TypeError(_type_error_str(val, np.ndarray))
        array_dict[key] = val
    return array_dict


class HdfEventReader(base.EventReaderBase):
    """A heparchy event reader object.

    :group: hepread

    Attributes
    ----------
    name : str
        The string representation of the event's name.
    count : int
        The number of particles within the event.
    edges : numpy.ndarray[int]
        A structured array providing edges for a graph representation,
        with fields "in" and "out".
    edge_weights : numpy.ndarray[double]
        Weights attributed to each edge.
    pmu : numpy.ndarray[double]
        A structured array providing particle 4-momenta, with fields
        "x", "y", "z", "e"
    color : numpy.ndarray[int]
        A structured array with fields "color" and "anti-color".
    pdg : numpy.ndarray[int]
        PDG codes providing the identity of every particle in the event.
    status : numpy.ndarray[int]
        Status codes, describing the reason for the generation of each
        particle in the event.
    helicity : numpy.ndarray[int]
        Spin polarisations of every particle in the event.
    final : numpy.ndarray[bool], deprecated
        Boolean mask over the data identifying particles in final state.
        Deprecated, simply use masks attribute with "final" subscript.
    available : list[str]
        Names of the datasets stored within the event.
    masks : MapReader[numpy.ndarray[bool]]
        Read-only dictionary-like interface to access boolean masks over
        the particle datasets.
    custom : MapReader[numpy.ndarray]
        Read-only dictionary-like interface to access user-defined
        custom datasets.
    custom_meta : MapReader
        Read-only dictionary-like interface to access user-defined
        metadata on the event.
    """

    def __init__(self) -> None:
        self.__name: ty.Optional[str] = None
        self.__grp: ty.Optional[Group] = None
        self._custom_grp: Group
        self._mask_grp: Group

    @property
    def _name(self) -> str:
        if self.__name is None:
            raise _NO_EVENT_ERR
        return self.__name

    @_name.setter
    def _name(self, val: str) -> None:
        self.__name = val

    @property
    def _grp(self) -> Group:
        if self.__grp is None:
            raise _NO_EVENT_ERR
        return self.__grp

    @_grp.setter
    def _grp(self, val: Group) -> None:
        self.__grp = val
        custom = val["custom"]
        mask = val["masks"]
        if (not isinstance(custom, Group)) or (not isinstance(mask, Group)):
            raise ValueError("Group not found")
        self._custom_grp = custom
        self._mask_grp = mask

    @property
    def name(self) -> str:
        return str(basename(self._name))

    @property
    def count(self) -> int:
        value = self._grp.attrs["num_pcls"]
        if not isinstance(value, np.integer):
            raise _NOT_NUMPY_ERR
        return int(value)

    @property  # type: ignore
    @_reg_event_builtin
    def edges(self) -> base.VoidVector:
        data = self._grp["edges"]
        if not isinstance(data, Dataset):
            raise ValueError("Stored data is corrupted")
        return data[...]

    @property  # type: ignore
    @_reg_event_builtin
    def edge_weights(self) -> base.DoubleVector:
        data = self._grp["edge_weights"]
        if not isinstance(data, Dataset):
            raise _NOT_NUMPY_ERR
        return data[...]

    @property  # type: ignore
    @_reg_event_builtin
    def pmu(self) -> base.VoidVector:
        data = self._grp["pmu"]
        if not isinstance(data, Dataset):
            raise _NOT_NUMPY_ERR
        return data[...]

    @property  # type: ignore
    @_reg_event_builtin
    def color(self) -> base.VoidVector:
        data = self._grp["color"]
        if not isinstance(data, Dataset):
            raise _NOT_NUMPY_ERR
        return data[...]

    @property  # type: ignore
    @_reg_event_builtin
    def pdg(self) -> base.IntVector:
        data = self._grp["pdg"]
        if not isinstance(data, Dataset):
            raise _NOT_NUMPY_ERR
        return data[...]

    @property  # type: ignore
    @_reg_event_builtin
    def status(self) -> base.HalfIntVector:
        data = self._grp["status"]
        if not isinstance(data, Dataset):
            raise _NOT_NUMPY_ERR
        return data[...]

    @property  # type: ignore
    @_reg_event_builtin
    def helicity(self) -> base.HalfIntVector:
        data = self._grp["helicity"]
        if not isinstance(data, Dataset):
            raise _NOT_NUMPY_ERR
        return data[...]

    @property  # type: ignore
    @deprecated
    def final(self) -> base.BoolVector:
        return self.masks["final"]

    @property
    def available(self) -> ty.List[str]:
        dataset_names = list()
        self._grp.visit(lambda name: dataset_names.append(name))
        return dataset_names

    @deprecated
    def mask(self, name: str) -> base.BoolVector:
        """Returns a boolean mask stored with the provided name."""
        return self.masks[name]

    @cached_property
    def masks(self) -> MapReader[base.BoolVector]:
        return MapReader[base.BoolVector](self, "_mask_grp", _grp_key_iter)

    @deprecated
    def get_custom(self, name: str) -> base.AnyVector:
        """Returns a custom dataset stored with the provided name."""
        return self.custom[name]

    @cached_property
    def custom(self) -> MapReader[base.AnyVector]:
        return MapReader[base.AnyVector](self, "_custom_grp", _grp_key_iter)

    @deprecated
    def get_custom_meta(self, name: str) -> ty.Any:
        """Returns custom metadata stored with the provided name."""
        return self._grp.attrs[name]

    @cached_property
    def custom_meta(self) -> MapReader[ty.Any]:
        return MapReader[ty.Any](self, "_grp", _meta_iter)

    def copy(self) -> HdfEventReader:
        """Returns a deep copy of the event object."""
        new_evt = self.__class__()
        new_evt._name = self._name
        new_evt._grp = self._grp
        return new_evt


class HdfProcessReader(base.ProcessReaderBase):
    """A heparchy process object. This is a numerically subscriptable
    iterator.

    :group: hepread

    Parameters
    ----------
    file_obj : HdfReader
        Instantiated heparchy file object.
    key : str
        The name of the process to be opened.

    Attributes
    ----------
    process_string : str
        The MadGraph string representation of the process.
    string : str, deprecated
        Deprecated, use process_string instead.
    signal_pdgs : numpy.ndarray[int]
        The PDG codes in the hard process which are considered signal.
    decay : dict[str, numpy.ndarray[int]], deprecated
        Provides access to incoming and outgoing parton PDG codes in
        the process. This has been deprecated due to the complexity
        introduced with intermediate states. Consider using signal_pdgs
        instead.
    com_energy : tuple[float, str]
        The magnitude and unit of the collision energy.
    custom_meta : MapReader
        Dictionary-like access to user-stored metadata.

    Examples
    --------
    Access a process named "higgs" and access events:

        >>> with heparchy.read.HdfReader("showers.hdf5") as hep_file:
        >>>     proc = hep_file["higgs"]
        >>>     for event in proc:  # iterate with for loop
        >>>         ...
        >>>     event = proc[5]  # 6th event with explicit indexing

    Read metadata stored in a process:

        >>> with heparchy.read.HdfReader("showers.hdf5") as hep_file:
        >>>     proc = hep_file["higgs"]
        >>>     print(proc.process_string)
        >>>     print(proc.signal_pdgs)
        >>>     print(proc.com_energy)
        >>>     print(proc.custom_meta["decay_channels"])
            p p > h z , (h > b b~) , (z > l+ l-)
            [25, 5, -5]
            (1300.0, "GeV")
            semileptonic

    Notes
    -----
    When implicitly iterating over this object (as in a for loop), the
    order of event retrieval will _generally_ follow the numerical order
    of the events, but this is not guaranteed. The iterator is based on
    HDF5's underlying B-tree structure, and will iterate in
    _native order_, _ie._ events will be read as quickly as possible.
    Native ordering does not change unless the file is modified, so the
    implicit iterator is consistent. However, if you require data to be
    retrieved in numerical order, rather than native order, use explicit
    indexing (as in the example above).
    """

    def __init__(self, file_obj: HdfReader, key: str) -> None:
        self._evt = HdfEventReader()
        grp = file_obj._buffer[key]
        if not isinstance(grp, Group):
            raise KeyError(f"{key} is not a process")
        self._grp: Group = grp
        self._meta: MetaDictType = dict(grp.attrs)
        evts_per_chunk = file_obj._buffer.attrs["evts_per_chunk"]
        if not isinstance(evts_per_chunk, np.integer):
            raise ValueError("Error reading metadata.")
        self._evts_per_chunk = int(evts_per_chunk)
        self.custom_meta: MapReader[ty.Any] = MapReader(self, "_grp", _meta_iter)

    def __len__(self) -> int:
        return int(self._meta["num_evts"])

    def __iter__(self) -> ty.Iterator[HdfEventReader]:
        for evt_chunk in self._grp.values():
            for name, evt in evt_chunk.items():
                self._evt._name = name
                self._evt._grp = evt
                yield self._evt

    def __getitem__(self, evt_num: int) -> HdfEventReader:
        chunk_idx, evt_idx = divmod(evt_num, self._evts_per_chunk)
        evt_name = event_key_format(evt_idx, self._evts_per_chunk)
        chunk_name = chunk_key_format(chunk_idx)
        self._evt._name = evt_name
        evt_chunk = self._grp[chunk_name]
        if not isinstance(evt_chunk, Group):
            raise ValueError
        evt_grp = evt_chunk[evt_name]
        if not isinstance(evt_grp, Group):
            raise ValueError
        self._evt._grp = evt_grp
        return self._evt

    @deprecated
    def read_event(self, evt_num: int) -> HdfEventReader:
        """Returns the event indexed by the passed index."""
        return self[evt_num]

    @property  # type: ignore
    @deprecated
    def string(self) -> str:
        return self.process_string

    @property
    def process_string(self) -> str:
        return _check_type(self._meta["process"], str)

    @property  # type: ignore
    @deprecated
    def decay(self) -> ty.Dict[str, base.IntVector]:
        return _format_array_dict(("in_pcls", "out_pcls"), self._meta)

    @property
    def signal_pdgs(self) -> base.IntVector:
        return _check_type(self._meta["signal_pdgs"], np.ndarray)

    @property
    def com_energy(self) -> ty.Tuple[float, str]:
        return (
            _check_type(self._meta["com_e"], float),
            _check_type(self._meta["e_unit"], str),
        )

    @deprecated
    def get_custom_meta(self, name: str) -> ty.Any:
        """Returns user-defined metadata stored under the give name."""
        return self._meta[name]


class HdfReader(base.ReaderBase):
    """Create a new heparchy hdf5 file object with read access.
    Processes stored within are accessed via string subscripting.

    :group: hepread

    Parameters
    ----------
    path : pathlib.Path
        Location of the file to be read.

    Examples
    --------
    Opening a file using the context manager:

        >>> import heparchy
        >>> with heparchy.read.HdfReader("showers.hdf5") as hep_file:
        >>>     ...

    """

    def __init__(self, path: ty.Union[Path, str]) -> None:
        self.path = Path(path)

    def __enter__(self) -> HdfReader:
        self._buffer = h5py.File(self.path, "r")
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self._buffer.close()

    def __getitem__(self, key: str) -> HdfProcessReader:
        return HdfProcessReader(self, key=key)

    @deprecated
    def read_process(self, name: str) -> HdfProcessReader:
        return self[name]
