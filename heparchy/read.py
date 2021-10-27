from functools import wraps
from copy import deepcopy
from os.path import basename

import numpy as np
import h5py

from heparchy import event_key_format


class HepReader:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self._buffer = h5py.File(self.path, 'r')
        return self

    def __len__(self):
        return self._buffer[self.key].attrs['num_evts']

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._buffer.close()

    def read_process(self, name: str):
        return self.ProcessReader(self._buffer, key=name)

    class ProcessReader:
        def __init__(self, file_obj, key: str):
            self.__buffer = file_obj
            self.key = key
            self.__evt = self.__EventReader(evt_data=(None, None))

        def __enter__(self):
            self.__proc_grp = self.__buffer[self.key]
            self._meta = dict(self.__buffer[self.key].attrs)
            return self

        def __exit__(self, exc_type, exc_value, exc_traceback):
            pass

        def __iter__(self):
            def iter_evts():
                for evt_data in self.__proc_grp.items():
                    self.__evt._name, self.__evt._grp = evt_data
                    yield self.__evt
            self.__iter = iter_evts()
            return self

        def __next__(self):
            return next(self.__iter)

        def __len__(self):
            return int(self._meta['num_evts'])

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

        # @property
        # def signal_id(self) -> int:
        #     return int(self._meta['signal_pcl'])

        def get_custom(self, name: str):
            return self._meta[name]
        
        def read_event(self, evt_num):
            evt_name = event_key_format(evt_num)
            self.__evt._name = evt_name
            self.__evt._grp = self.__proc_grp[evt_name]
            return self.__evt

        class __EventReader:
            __slots__ = ('_name', '_grp')

            def __init__(self, evt_data):
                self._name = evt_data[0]
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
            def pmu(self) -> np.ndarray:
                return self._grp['pmu'][...]

            @property
            def pdg(self) -> np.ndarray:
                return self._grp['pdg'][...]

            @property
            def available(self) -> list:
                dataset_names = list()
                self._grp.visit(lambda name: dataset_names.append(name))
                return dataset_names

            def mask(self, name: str) -> np.ndarray:
                return self._grp[name][...]

            def get_custom(self, name: str):
                return self._grp[name][...]

            def copy(self):
                return deepcopy(self)
