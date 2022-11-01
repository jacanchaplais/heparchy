"""
`heparchy.read.hdf`
===================

Defines the data structures to open and iterate through heparchy
formatted HDF5 files.
"""
from __future__ import annotations
from typing import (
        Any, List, Dict, Sequence, Type, Iterator, Union, Set, TypeVar,
        Callable, Generic, Tuple)
from collections.abc import Mapping
from copy import deepcopy
from os.path import basename
from pathlib import Path
from functools import cached_property

import numpy as np
import h5py
from h5py import AttributeManager, Group, Dataset

from heparchy.utils import event_key_format, deprecated
from .base import ReaderBase, EventReaderBase, ProcessReaderBase
from heparchy.annotate import (
        IntVector, HalfIntVector, DoubleVector, BoolVector, AnyVector,
        DataType)


MetaDictType = Dict[str, Union[str, int, float, bool, AnyVector]]
READ_ONLY_MSG = "Attribute is read-only."
BUILTIN_PROPS: Set[str] = set()
BUILTIN_METADATA = {  # TODO: work out non-hardcoded
    "HdfEventReader": {"num_pcls", "mask_keys"},
    "HdfProcessReader": {"signal_pdgs", "process", "com_e", "e_unit"}
    }
PropMethod = TypeVar("PropMethod", bound=Callable)
MapValue = TypeVar("MapValue")
NOT_NUMPY_ERR = ValueError("Stored data type is corrupted.")
ReaderType = Union["HdfEventReader", "HdfProcessReader"]


def reg_event_builtin(func: PropMethod) -> PropMethod:
    BUILTIN_PROPS.add(func.__name__)
    return func


def stored_keys(attrs: AttributeManager, key_attr_name: str) -> Iterator[str]:
    key_ds = attrs[key_attr_name]
    if not isinstance(key_ds, np.ndarray):
        raise NOT_NUMPY_ERR
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
        yield from stored_keys(grp.attrs, key_attr_name)


def _custom_iter(reader: HdfEventReader) -> Iterator[str]:
    key_attr_name = "custom_keys"
    grp = reader._grp
    if key_attr_name not in grp.attrs:
        names = set(grp.keys()) - set(reader.masks.keys())
        for name in (names - BUILTIN_PROPS):
            yield name
    else:
        yield from stored_keys(grp.attrs, key_attr_name)


def _meta_iter(reader: ReaderType) -> Iterator[str]:
    key_attr_name = "custom_meta_keys"
    grp = reader._grp
    if key_attr_name not in grp.attrs:
        names = set(grp.attrs.keys())
        for name in (names - BUILTIN_METADATA[reader.__class__.__name__]):
            yield name
    else:
        yield from stored_keys(grp.attrs, key_attr_name)


class MapReader(Generic[MapValue], Mapping[str, MapValue]):
    """Read-only dictionary-like interface to user-named heparchy
    datasets.
    """
    def __init__(self,
                 reader: ReaderType,
                 map_iter: Callable[..., Iterator[str]]) -> None:
        self._reader = reader
        self._map_iter = map_iter

    def __repr__(self) -> str:
        dset_repr = "<Read-Only Data>"
        kv = ", ".join(map(lambda k: f"\'{k}\': {dset_repr}", self))
        return "{" + f"{kv}" + "}"

    def __len__(self) -> int:
        return len(tuple(iter(self)))

    def __getitem__(self, name: str) -> MapValue:
        if name not in set(self):
            raise KeyError("No data stored with this name")
        if self._map_iter.__name__ == "_meta_iter":
            return self._reader._grp.attrs[name]  # type: ignore
        data = self._reader._grp[name]
        if not isinstance(data, Dataset):
            raise NOT_NUMPY_ERR
        return data[...]  # type: ignore

    def __setitem__(self, name: str, data: MapValue) -> None:
        raise AttributeError(READ_ONLY_MSG)

    def __delitem__(self, name: str) -> None:
        raise AttributeError(READ_ONLY_MSG)

    def __iter__(self) -> Iterator[str]:
        yield from self._map_iter(self._reader)


def type_error_str(data: Any, dtype: type) -> str:
    return ("Type mismatch: retrieved value should be of "
            f"type {dtype}, found instance of {type(data)}")


def check_type(data: Any, dtype: Type[DataType]) -> DataType:
    if not isinstance(data, dtype):
        raise TypeError(type_error_str(data, dtype))
    return data


def format_array_dict(
        keys: Sequence[str], in_dict: Dict[str, Any]) -> Dict[str, AnyVector]:
    array_dict = dict()
    for key in keys:
        val = in_dict[key][...]
        if not isinstance(val, np.ndarray):
            raise TypeError(type_error_str(val, np.ndarray))
        array_dict[key] = val
    return array_dict


class HdfEventReader(EventReaderBase):
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
            raise NOT_NUMPY_ERR
        return int(value)

    @property  # type: ignore
    @reg_event_builtin
    def edges(self) -> AnyVector:
        data = self._grp["edges"]
        if not isinstance(data, Dataset):
            raise ValueError("Stored data is corrupted")
        return data[...]

    @property  # type: ignore
    @reg_event_builtin
    def edge_weights(self) -> DoubleVector:
        data = self._grp['edge_weights']
        if not isinstance(data, Dataset):
            raise NOT_NUMPY_ERR
        return data[...]

    @property  # type: ignore
    @reg_event_builtin
    def pmu(self) -> AnyVector:
        data = self._grp['pmu']
        if not isinstance(data, Dataset):
            raise NOT_NUMPY_ERR
        return data[...]

    @property  # type: ignore
    @reg_event_builtin
    def color(self) -> AnyVector:
        data = self._grp['color']
        if not isinstance(data, Dataset):
            raise NOT_NUMPY_ERR
        return data[...]

    @property  # type: ignore
    @reg_event_builtin
    def pdg(self) -> IntVector:
        data = self._grp['pdg']
        if not isinstance(data, Dataset):
            raise NOT_NUMPY_ERR
        return data[...]

    @property  # type: ignore
    @reg_event_builtin
    def status(self) -> HalfIntVector:
        data = self._grp['status']
        if not isinstance(data, Dataset):
            raise NOT_NUMPY_ERR
        return data[...]

    @property  # type: ignore
    @reg_event_builtin
    def helicity(self) -> HalfIntVector:
        data = self._grp['helicity']
        if not isinstance(data, Dataset):
            raise NOT_NUMPY_ERR
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
        return self.masks[name]

    @cached_property
    def masks(self) -> MapReader[BoolVector]:
        return MapReader[BoolVector](self, _mask_iter)

    @deprecated
    def get_custom(self, name: str) -> AnyVector:
        return self.custom[name]
    
    @cached_property
    def custom(self) -> MapReader[AnyVector]:
        return MapReader[AnyVector](self, _custom_iter)

    @deprecated
    def get_custom_meta(self, name: str) -> Any:
        return self._grp.attrs[name]

    @cached_property
    def custom_meta(self) -> MapReader[Any]:
        return MapReader[Any](self, _meta_iter)

    def copy(self) -> HdfEventReader:
        return deepcopy(self)


class HdfProcessReader(ProcessReaderBase):
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
        return self[evt_num]

    @property  # type: ignore
    @deprecated
    def string(self) -> str:
        return self.process_string

    @property
    def process_string(self) -> str:
        return check_type(self._meta["process"], str)

    @property  # type: ignore
    @deprecated
    def decay(self) -> Dict[str, IntVector]:
        return format_array_dict(("in_pcls", "out_pcls"), self._meta)

    @property
    def signal_pdgs(self) -> IntVector:
        return check_type(self._meta["signal_pdgs"], np.ndarray)

    @property
    def com_energy(self) -> Tuple[float, str]:
        return (check_type(self._meta['com_e'], float),
                check_type(self._meta['e_unit'], str))

    @deprecated
    def get_custom_meta(self, name: str) -> Any:
        return self._meta[name]


class HdfReader(ReaderBase):
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
