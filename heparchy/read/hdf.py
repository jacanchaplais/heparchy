"""
`heparchy.read.hdf`
===================

Defines the data structures to open and iterate through heparchy
formatted HDF5 files.
"""
from __future__ import annotations
from pathlib import Path
from os.path import basename
from copy import deepcopy
from functools import cached_property
from collections.abc import Mapping
from typing import (
        Any, List, Dict, Sequence, Type, Iterator, Union, Set, TypeVar,
        Callable, Generic, Tuple)

import numpy as np
import h5py
from h5py import AttributeManager, Group, Dataset

from heparchy.utils import event_key_format, deprecated
from .base import ReaderBase, EventReaderBase, ProcessReaderBase
from heparchy.annotate import (
        IntVector, HalfIntVector, DoubleVector, BoolVector, AnyVector,
        DataType)


__all__: List[str] = []


def _export(procedure: ExportType) -> ExportType:
    __all__.append(procedure.__name__)
    return procedure


_NOT_NUMPY_ERR = ValueError("Stored data type is corrupted.")
_BUILTIN_PROPS: Set[str] = set()
_BUILTIN_METADATA = {  # TODO: work out non-hardcoded
    "HdfEventReader": {"num_pcls", "mask_keys"},
    "HdfProcessReader": {"signal_pdgs", "process", "com_e", "e_unit"}
    }

MetaDictType = Dict[str, Union[str, int, float, bool, AnyVector]]
ReaderType = Union["HdfEventReader", "HdfProcessReader"]
ExportType = TypeVar("ExportType", Callable, Type)
PropMethod = TypeVar("PropMethod", bound=Callable)
MapValue = TypeVar("MapValue")


@_export
class ReadOnlyError(RuntimeError):
    """Raised when trying to write to read-only data."""


def _reg_event_builtin(func: PropMethod) -> PropMethod:
    _BUILTIN_PROPS.add(func.__name__)
    return func


def _stored_keys(attrs: AttributeManager, key_attr_name: str) -> Iterator[str]:
    key_ds = attrs[key_attr_name]
    if not isinstance(key_ds, np.ndarray):
        raise _NOT_NUMPY_ERR
    for name in tuple(key_ds):
        yield name


def _mask_iter(reader: HdfEventReader) -> Iterator[str]:
    key_attr_name = "mask_keys"
    grp = reader._grp
    if key_attr_name not in grp.attrs:
        dtype = np.dtype("<?")
        for name, dset in grp.items():
            if dset.dtype != dtype:
                continue
            yield name
    else:
        yield from _stored_keys(grp.attrs, key_attr_name)


def _custom_iter(reader: HdfEventReader) -> Iterator[str]:
    key_attr_name = "custom_keys"
    grp = reader._grp
    if key_attr_name not in grp.attrs:
        names = set(grp.keys()) - set(reader.masks.keys())
        for name in (names - _BUILTIN_PROPS):
            yield name
    else:
        yield from _stored_keys(grp.attrs, key_attr_name)


def _meta_iter(reader: ReaderType) -> Iterator[str]:
    key_attr_name = "custom_meta_keys"
    grp = reader._grp
    if key_attr_name not in grp.attrs:
        names = set(grp.attrs.keys())
        for name in (names - _BUILTIN_METADATA[reader.__class__.__name__]):
            yield name
    else:
        yield from _stored_keys(grp.attrs, key_attr_name)


@_export
class MapReader(Generic[MapValue], Mapping[str, MapValue]):
    """Read-only dictionary-like interface to user-named heparchy
    datasets.

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
    def __init__(self,
                 reader: ReaderType,
                 iter_func: Callable[..., Iterator[str]]) -> None:
        self._reader = reader
        self._iter_func = iter_func

    def __repr__(self) -> str:
        dset_repr = "<Read-Only Data>"
        kv = ", ".join(map(lambda k: f"\'{k}\': {dset_repr}", self))
        return "{" + f"{kv}" + "}"

    def __len__(self) -> int:
        return len(tuple(iter(self)))

    def __getitem__(self, name: str) -> MapValue:
        if name not in set(self):
            raise KeyError("No data stored with this name")
        if self._iter_func.__name__ == "_meta_iter":
            return self._reader._grp.attrs[name]  # type: ignore
        data = self._reader._grp[name]
        if not isinstance(data, Dataset):
            raise _NOT_NUMPY_ERR
        return data[...]  # type: ignore

    def __setitem__(self, name: str, data: MapValue) -> None:
        raise ReadOnlyError("Value is read-only")

    def __delitem__(self, name: str) -> None:
        raise ReadOnlyError("Value is read-only")

    def __iter__(self) -> Iterator[str]:
        yield from self._iter_func(self._reader)


def _type_error_str(data: Any, dtype: type) -> str:
    return ("Type mismatch: retrieved value should be of "
            f"type {dtype}, found instance of {type(data)}")


def _check_type(data: Any, dtype: Type[DataType]) -> DataType:
    if not isinstance(data, dtype):
        raise TypeError(_type_error_str(data, dtype))
    return data


def _format_array_dict(
        keys: Sequence[str], in_dict: Dict[str, Any]) -> Dict[str, AnyVector]:
    array_dict = dict()
    for key in keys:
        val = in_dict[key][...]
        if not isinstance(val, np.ndarray):
            raise TypeError(_type_error_str(val, np.ndarray))
        array_dict[key] = val
    return array_dict


@_export
class HdfEventReader(EventReaderBase):
    """A heparchy event reader object.

    Attributes (read-only)
    ----------------------
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
    __slots__ = ('_name', '_grp')

    def __init__(self, evt_data) -> None:
        self._name: str = evt_data[0]
        self._grp: Group = evt_data[1]

    @property
    def name(self) -> str:
        return str(basename(self._name))

    @property
    def count(self) -> int:
        value = self._grp.attrs['num_pcls']
        if not isinstance(value, np.integer):
            raise _NOT_NUMPY_ERR
        return int(value)

    @property  # type: ignore
    @_reg_event_builtin
    def edges(self) -> AnyVector:
        data = self._grp["edges"]
        if not isinstance(data, Dataset):
            raise ValueError("Stored data is corrupted")
        return data[...]

    @property  # type: ignore
    @_reg_event_builtin
    def edge_weights(self) -> DoubleVector:
        data = self._grp['edge_weights']
        if not isinstance(data, Dataset):
            raise _NOT_NUMPY_ERR
        return data[...]

    @property  # type: ignore
    @_reg_event_builtin
    def pmu(self) -> AnyVector:
        data = self._grp['pmu']
        if not isinstance(data, Dataset):
            raise _NOT_NUMPY_ERR
        return data[...]

    @property  # type: ignore
    @_reg_event_builtin
    def color(self) -> AnyVector:
        data = self._grp['color']
        if not isinstance(data, Dataset):
            raise _NOT_NUMPY_ERR
        return data[...]

    @property  # type: ignore
    @_reg_event_builtin
    def pdg(self) -> IntVector:
        data = self._grp['pdg']
        if not isinstance(data, Dataset):
            raise _NOT_NUMPY_ERR
        return data[...]

    @property  # type: ignore
    @_reg_event_builtin
    def status(self) -> HalfIntVector:
        data = self._grp['status']
        if not isinstance(data, Dataset):
            raise _NOT_NUMPY_ERR
        return data[...]

    @property  # type: ignore
    @_reg_event_builtin
    def helicity(self) -> HalfIntVector:
        data = self._grp['helicity']
        if not isinstance(data, Dataset):
            raise _NOT_NUMPY_ERR
        return data[...]

    @property  # type: ignore
    @deprecated
    def final(self) -> BoolVector:
        return self.masks["final"]

    @property
    def available(self) -> List[str]:
        dataset_names = list()
        self._grp.visit(lambda name: dataset_names.append(name))
        return dataset_names

    @deprecated
    def mask(self, name: str) -> BoolVector:
        """Returns a boolean mask stored with the provided name."""
        return self.masks[name]

    @cached_property
    def masks(self) -> MapReader[BoolVector]:
        return MapReader[BoolVector](self, _mask_iter)

    @deprecated
    def get_custom(self, name: str) -> AnyVector:
        """Returns a custom dataset stored with the provided name."""
        return self.custom[name]
    
    @cached_property
    def custom(self) -> MapReader[AnyVector]:
        return MapReader[AnyVector](self, _custom_iter)

    @deprecated
    def get_custom_meta(self, name: str) -> Any:
        """Returns custom metadata stored with the provided name."""
        return self._grp.attrs[name]

    @cached_property
    def custom_meta(self) -> MapReader[Any]:
        return MapReader[Any](self, _meta_iter)

    def copy(self) -> HdfEventReader:
        """Returns a deep copy of the event object."""
        return deepcopy(self)


@_export
class HdfProcessReader(ProcessReaderBase):
    """A heparchy process object. This is a numerically subscriptable
    iterator.

    Parameters
    ----------
    file_obj : HdfReader
        Instantiated heparchy file object.
    key : str
        The name of the process to be opened.

    Attributes (read-only)
    ----------------------
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
    >>>     event = proc[5]  # access 6th event with explicit indexing

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
    
    """
    def __init__(self, file_obj: HdfReader, key: str) -> None:
        self._evt = HdfEventReader(evt_data=(None, None))
        grp = file_obj._buffer[key]
        if not isinstance(grp, Group):
            raise KeyError(f"{key} is not a process")
        self._grp: Group = grp
        self._meta: MetaDictType = dict(file_obj._buffer[key].attrs)
        self.custom_meta: MapReader[Any] = MapReader(self, _meta_iter)

    def __len__(self) -> int:
        return int(self._meta["num_evts"])

    def __iter__(self) -> Iterator[HdfEventReader]:
        self._evt_gen = (self[i] for i in range(len(self)))
        return self

    def __next__(self) -> HdfEventReader:
        return next(self._evt_gen)

    def __getitem__(self, evt_num: int) -> HdfEventReader:
        evt_name = event_key_format(evt_num)
        self._evt._name = evt_name
        evt_grp = self._grp[evt_name]
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
    def decay(self) -> Dict[str, IntVector]:
        return _format_array_dict(("in_pcls", "out_pcls"), self._meta)

    @property
    def signal_pdgs(self) -> IntVector:
        return _check_type(self._meta["signal_pdgs"], np.ndarray)

    @property
    def com_energy(self) -> Tuple[float, str]:
        return (_check_type(self._meta['com_e'], float),
                _check_type(self._meta['e_unit'], str))

    @deprecated
    def get_custom_meta(self, name: str) -> Any:
        """Returns user-defined metadata stored under the give name."""
        return self._meta[name]


@_export
class HdfReader(ReaderBase):
    """Create a new heparchy hdf5 file object with read access.
    Processes stored within are accessed via string subscripting.

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
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> HdfReader:
        self._buffer = h5py.File(self.path, 'r')
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self._buffer.close()

    def __getitem__(self, key: str) -> HdfProcessReader:
        return HdfProcessReader(self, key=key)

    @deprecated
    def read_process(self, name: str) -> HdfProcessReader:
        return self[name]
