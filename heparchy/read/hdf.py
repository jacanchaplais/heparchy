"""
`heparchy.read.hdf`
===================

Defines the data structures to open and iterate through heparchy
formatted HDF5 files.
"""
from __future__ import annotations
from typing import Any, List, Dict, Sequence, Type, Iterator, Union, Set
from collections.abc import Mapping
from copy import deepcopy
from os.path import basename
from pathlib import Path
from functools import cached_property

import numpy as np
import h5py
from h5py import Group

from heparchy.utils import event_key_format, deprecated
from .base import (
        ReaderBase, EventReaderBase, ProcessReaderBase, ComEnergyType)
from heparchy.annotate import (
        IntVector, HalfIntVector, DoubleVector, BoolVector, AnyVector,
        DataType)


MetaDictType = Dict[str, Union[str, int, float, bool, AnyVector]]
READ_ONLY_MSG = "Attribute is read-only."


class MaskReader(Mapping[str, BoolVector]):
    def __init__(self, event: HdfEventReader) -> None:
        self.event = event

    def __repr__(self) -> str:
        dset_repr = "<Read-Only Data>"
        kv = ", ".join(map(lambda k: f"\'{k}\': {dset_repr}", self))
        return "{" + f"{kv}" + "}"

    def __len__(self) -> int:
        return len(tuple(iter(self)))

    def __getitem__(self, name: str) -> BoolVector:
        if name not in set(self):
            raise KeyError("No mask stored with this name")
        return self.event._grp[name][...]

    def __setitem__(self, name: str, data: BoolVector) -> None:
        raise AttributeError(READ_ONLY_MSG)

    def __delitem__(self, name: str) -> None:
        raise AttributeError(READ_ONLY_MSG)

    def __iter__(self) -> Iterator[str]:
        mask_keys = "mask_keys"
        grp = self.event._grp
        if mask_keys not in grp.attrs:
            dtype = np.dtype("<?")
            for name, dset in grp.items():
                if name == "final" or dset.dtype != dtype:
                    continue
                yield name
        else:
            for name in grp.attrs[mask_keys]:
                yield name


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
        self._grp = evt_data[1]

    @property
    def name(self) -> str:
        return str(basename(self._name))

    @property
    def count(self) -> int:
        return int(self._grp.attrs['num_pcls'])

    @property
    def edges(self) -> AnyVector:
        return self._grp['edges'][...]

    @property
    def edge_weights(self) -> DoubleVector:
        return self._grp['edge_weights'][...]

    @property
    def pmu(self) -> AnyVector:
        return self._grp['pmu'][...]

    @property
    def color(self) -> AnyVector:
        return self._grp['color'][...]

    @property
    def pdg(self) -> IntVector:
        return self._grp['pdg'][...]

    @property
    def status(self) -> HalfIntVector:
        return self._grp['status'][...]

    @property
    def helicity(self) -> HalfIntVector:
        return self._grp['helicity'][...]

    @property
    def final(self) -> BoolVector:
        return self.mask('final')

    @property
    def available(self) -> List[str]:
        dataset_names = list()
        self._grp.visit(lambda name: dataset_names.append(name))
        return dataset_names

    @deprecated
    def mask(self, name: str) -> BoolVector:
        return self._grp[name][...]

    @cached_property
    def masks(self) -> MaskReader:
        return MaskReader(self)

    def get_custom(self, name: str) -> AnyVector:
        return self._grp[name][...]

    def get_custom_meta(self, name: str) -> Any:
        return self._grp.attrs[name]
    
    def copy(self) -> HdfEventReader:
        return deepcopy(self)


class HdfProcessReader(ProcessReaderBase):
    def __init__(self, file_obj: HdfReader, key: str) -> None:
        self.__evt = HdfEventReader(evt_data=(None, None))
        grp = file_obj._buffer[key]
        if not isinstance(grp, Group):
            raise KeyError(f"{key} is not a process")
        self.__proc_grp = grp
        self._meta: MetaDictType = dict(file_obj._buffer[key].attrs)

    def __len__(self) -> int:
        return int(self._meta["num_evts"])

    def __iter__(self) -> Iterator[HdfEventReader]:
        self._evt_gen = (self[i] for i in range(len(self)))
        return self

    def __next__(self) -> HdfEventReader:
        return next(self._evt_gen)

    def __getitem__(self, evt_num: int) -> HdfEventReader:
        evt_name = event_key_format(evt_num)
        self.__evt._name = evt_name
        self.__evt._grp = self.__proc_grp[evt_name]
        return self.__evt

    @property
    def meta(self) -> MetaDictType:
        return self._meta

    @deprecated
    def read_event(self, evt_num: int) -> HdfEventReader:
        return self[evt_num]

    @property
    def string(self) -> str:
        return check_type(self._meta["process"], str)

    @property
    def decay(self) -> Dict[str, IntVector]:
        return format_array_dict(("in_pcls", "out_pcls"), self._meta)

    @property
    def com_energy(self) -> ComEnergyType:
        return {
            "energy": check_type(self._meta['com_e'], float),
            "unit": check_type(self._meta['e_unit'], str),
            }

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
