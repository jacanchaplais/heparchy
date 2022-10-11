from copy import deepcopy
from os.path import basename
from warnings import warn

import numpy as np
import h5py

from heparchy import event_key_format
from ._base import ReaderBase, EventReaderBase, ProcessReaderBase


class _EventReader(EventReaderBase):
    __slots__ = ('_name', '_grp')

    def __init__(self, evt_data):
        self._name: str = evt_data[0]
        self._grp = evt_data[1]

    @property
    def name(self) -> str:
        return str(basename(self._name))

    @property
    def count(self) -> int:
        return int(self._grp.attrs['num_pcls'])

    @property
    def edges(self) -> np.ndarray:
        return self._grp['edges'][...]

    @property
    def edge_weights(self) -> np.ndarray:
        return self._grp['edge_weights'][...]

    @property
    def pmu(self) -> np.ndarray:
        return self._grp['pmu'][...]

    @property
    def color(self) -> np.ndarray:
        return self._grp['color'][...]

    @property
    def pdg(self) -> np.ndarray:
        return self._grp['pdg'][...]

    @property
    def status(self) -> np.ndarray:
        return self._grp['status'][...]

    @property
    def helicity(self) -> np.ndarray:
        return self._grp['helicity'][...]

    @property
    def final(self) -> np.ndarray:
        return self.mask('final')

    @property
    def available(self) -> list:
        dataset_names = list()
        self._grp.visit(lambda name: dataset_names.append(name))
        return dataset_names

    def mask(self, name: str) -> np.ndarray:
        return self._grp[name][...]

    def get_custom(self, name: str):
        return self._grp[name][...]

    def get_custom_meta(self, name: str):
        return self._grp.attrs[name]
    
    def copy(self):
        return deepcopy(self)


class _ProcessReader(ProcessReaderBase):
    def __init__(self, file_obj, key: str):
        self.__evt = _EventReader(evt_data=(None, None))
        self.__proc_grp: h5py.Group = file_obj._buffer[key]
        self._meta = dict(file_obj._buffer[key].attrs)

    def __len__(self) -> int:
        return int(self._meta['num_evts'])

    def __iter__(self):
        self._evt_gen = (self[i] for i in range(len(self)))
        return self

    def __next__(self) -> _EventReader:
        return next(self._evt_gen)

    def __getitem__(self, evt_num: int) -> _EventReader:
        evt_name = event_key_format(evt_num)
        self.__evt._name = evt_name
        self.__evt._grp = self.__proc_grp[evt_name]
        return self.__evt

    def read_event(self, evt_num: int) -> _EventReader:
        warn("read_event is deprecated, select events by directly "
             "subscripting the process object, eg.\n"
             "event = proc[0]\n"
             "or iterating over the process with a for-loop, eg.\n"
             "for event in proc: ...\n"
             "This method and notice will be removed in version 1.0.0.",
             DeprecationWarning, stacklevel=2)
        return self[evt_num]

    @property
    def string(self) -> str:
        return self._meta['process']

    @property
    def decay(self) -> dict:
        return {
            'in_pcls': self._meta['in_pcls'],
            'out_pcls': self._meta['out_pcls'],
            }

    @property
    def com_energy(self) -> dict:
        return {
            'energy': self._meta['com_e'],
            'unit': self._meta['e_unit'],
            }

    def get_custom_meta(self, name: str):
        return self._meta[name]


class HdfReader(ReaderBase):
    def __init__(self, path: str):
        self.path = path

    def __enter__(self):
        self._buffer = h5py.File(self.path, 'r')
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._buffer.close()

    def read_process(self, name: str):
        return _ProcessReader(self, key=name)
