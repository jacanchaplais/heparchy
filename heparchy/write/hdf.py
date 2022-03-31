from typing import Any, Optional

import numpy as np
import h5py

from heparchy import event_key_format
from ._base import WriterBase, ProcessWriterBase, EventWriterBase


class _EventWriter(EventWriterBase):
    def __init__(self, grp_obj: ProcessWriterBase):
        from typicle import Types
        self.__types = Types()
        self.__grp_obj = grp_obj # pointer to parent group obj
        self._idx = grp_obj._evt_idx # index for current event
        self.num_pcls = None

    def __enter__(self):
        self.__evt = self.__grp_obj._grp.create_group(
                event_key_format(self._idx))
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # increasing the index for the group to create next event
        # also acts as running counter for number events
        if self.num_pcls == None:
            self.num_pcls = 0
        self.__evt.attrs['num_pcls'] = self.num_pcls
        self.__grp_obj._evt_idx += 1

    def __set_num_pcls(self, data: np.ndarray):
        shape = data.shape
        num_pcls = shape[0]

        if self.num_pcls == None:
            self.num_pcls = num_pcls
        elif num_pcls != self.num_pcls:
            raise ValueError(
                    f'Datasets within same event must have the same '
                    + f'number of particles (rows, if 2D array). '
                    + f'Previous datasets had {self.num_pcls}, '
                    + f'attempted dataset would have {num_pcls}.'
                    )
        else:
            return

    def __mk_dset(
            self, name: str, data: np.ndarray, shape: tuple,
            dtype: Any) -> None:
        """Generic dataset creation and population function.
        Wrap in methods exposed to the user interface.
        """
        # check data can be broadcasted to dataset:
        if data.squeeze().shape != shape:
            raise ValueError(
                    f'Input data shape {data.shape} '
                    + f'incompatible with dataset shape {shape}.'
                    )
        dset = self.__evt.create_dataset(
                name=name,
                shape=shape,
                dtype=dtype,
                shuffle=True,
                compression='lzf',
                )
        dset[...] = data

    def set_edges(
            self,
            data: np.ndarray,
            weights: Optional[np.ndarray] = None,
            strict_size: bool = True) -> None:
        """Write edge indices for event.
        
        Parameters
        ----------
        data : 2d array of ints
            Each row contains a pair of in / out edge indices.
        strict_size : bool
            If True will assume number of edges = number of particles
            (default: True).
        """
        if strict_size is True:
            self.__set_num_pcls(data)
        if weights is not None:
            num_edges = len(data)
            num_weights = len(weights)
            if num_edges != num_weights:
                raise ValueError(
                    f"Incompatible shapes. Number of edges = {num_edges} and "
                    + f"number of weights = {num_weights}. Must be same size."
                    )
            self.__mk_dset(
                    name='edge_weights',
                    data=weights,
                    shape=weights.shape,
                    dtype=self.__types.double,
                    )
        self.__mk_dset(
                name='edges',
                data=data,
                shape=data.shape,
                dtype=self.__types.edge,
                )

    def set_pmu(self, data: np.ndarray) -> None:
        """Write 4-momentum for all particles to event.
        
        Parameters
        ----------
        data : 2d array of floats.
            Each row contains momentum of one particle,
            in order [px, py, pz, e].
        """
        self.__set_num_pcls(data)
        self.__mk_dset(
                name='pmu',
                data=data,
                shape=data.shape,
                dtype=self.__types.pmu,
                )

    def set_color(self, data: np.ndarray) -> None:
        """Write color / anticolor pairs for all particles to event.
        
        Parameters
        ----------
        data : 2d array of ints.
            Each row contains color / anticolor values respectively.
        """
        self.__set_num_pcls(data)
        self.__mk_dset(
                name='color',
                data=data,
                shape=data.shape,
                dtype=self.__types.color,
                )

    def set_pdg(self, data: np.ndarray) -> None:
        """Write pdg codes for all particles to event.
        
        Parameters
        ----------
        data : iterable or 1d numpy array
            Iterable of ints containing pdg codes for every
            particle in the event.
        """
        self.__set_num_pcls(data)
        self.__mk_dset(
                name='pdg',
                data=data,
                shape=(self.num_pcls,),
                dtype=self.__types.int,
                )

    def set_status(self, data: np.ndarray) -> None:
        """Write status codes for all particles to event.
        
        Parameters
        ----------
        data : iterable or 1d numpy array
            Iterable of ints containing status codes for every
            particle in the event.
        """
        self.__set_num_pcls(data)
        self.__mk_dset(
                name='status',
                data=data,
                shape=(self.num_pcls,),
                dtype=self.__types.int,
                )

    def set_helicity(self, data: np.ndarray) -> None:
        """Write helicity values for all particles to event.

        Parameters
        ----------
        data : iterable or 1d numpy array
            Iterable of floats containing helicity values for every
            particle in the event.
        """
        self.__set_num_pcls(data)
        self.__mk_dset(
                name='helicity',
                data=data,
                shape=(self.num_pcls,),
                dtype=self.__types.helicity,
                )
    
    
    def set_mask(self, name: str, data: np.ndarray) -> None:
        """Write bool mask for all particles in event.
        
        Parameters
        ----------
        data : iterable or 1d numpy array
            Iterable of bools containing True / False values for
            every particle in event.
            Note: would also accept int iterable of 0s / 1s.

        Notes
        -----
        Example use cases:
            - identify particles from specific parent
            - provide mask for rapidity and pT cuts
            - if storing whole shower, identify final state
        """
        self.__set_num_pcls(data)
        self.__mk_dset(
                name=name,
                data=data,
                shape=(self.num_pcls,),
                dtype=self.__types.bool,
                )

    def set_custom_dataset(
            self, name: str, data: np.ndarray, dtype: Any) -> None:
        """Write a custom dataset to the event.

        Parameters
        ----------
        name : str
            Handle used when reading the data.
        data : nd numpy array
            data to store.
        dtype : valid string, numpy, or python data type
            Type in which your data should be encoded for
            storage.
            Note: using little Endian for builtin datasets.
        """
        self.__set_num_pcls(data)
        self.__mk_dset(
                name=name,
                data=data,
                shape=data.shape,
                dtype=dtype,
                )

    def set_custom_meta(self, name: str, metadata: Any) -> None:
        """Store custom metadata to the event.

        Parameters
        ----------
        name : str
            Handle to access the metadata at read time.
        metadata : str, int, float, or iterables thereof
            The data you wish to store.
        """
        self.__evt.attrs[name] = metadata


class _ProcessWriter(ProcessWriterBase):
    def __init__(self, file_obj: WriterBase, key: str):
        self.__file_obj = file_obj
        self.key = key
        self._evt_idx = 0

    def __enter__(self):
        self._grp = self.__file_obj._buffer.create_group(self.key)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # count all of the events and write to attribute
        self._grp.attrs['num_evts'] = self._evt_idx

    def set_string(self, proc_str: str) -> None:
        """Writes the string formatted underlying event to the
        process metadata.
        
        Parameters
        ----------
        proc_str : str
            MadGraph formatted string representing the hard event,
            eg. p p > t t~
        """
        self._grp.attrs['process'] = proc_str

    def set_decay(self, in_pcls: tuple, out_pcls: tuple) -> None:
        """Writes the pdgids of incoming and outgoing particles to
        process metadata.

        Parameters
        ----------
        in_pcls : tuple of ints
            pdgids of incoming particles.
            eg. for p p => (2212, 2212)
        out_pcls : tuple of ints
            pdgids of outgoing particles.
            eg. for t t~ => (6, -6)
        """
        self._grp.attrs['in_pcls'] = in_pcls
        self._grp.attrs['out_pcls'] = out_pcls

    def set_com_energy(self, energy: float, unit: str) -> None:
        """Writes the CoM energy and unit for the collision to
        process metadata.

        Parameters
        ----------
        energy : float
            eg. 13.0
        out_pcls : str
            eg. 'GeV'
        """
        self._grp.attrs['com_e'] = energy
        self._grp.attrs['e_unit'] = unit

    def set_custom_meta(self, name: str, metadata: Any) -> None:
        """Store custom metadata to the process.

        Parameters
        ----------
        name : str
            Handle to access the metadata at read time.
        metadata : str, int, float, or iterables thereof
            The data you wish to store.
        """
        self._grp.attrs[name] = metadata

    def new_event(self) -> _EventWriter:
        return _EventWriter(self)

class HdfWriter(WriterBase):
    """Class provides nested context managers for handling writing
    particle data for hierarchical access.

    The outer context manager (instantiated by this class) constructs
    and prepares the file for writing.

    The next layer down, a process context manager can be instantiated
    with the `new_process()` method. This acts as a container for
    hadronised showers generated with the same underlying process.
    It may be called repeatedly to store data from different processes
    within the same file.
    Optional methods are provided to write metadata to these containers,
    describing the hard event.
    The number of events written within this container is automatically
    stored to the metadata.

    The final layer is the event context manager, and can be
    instantiated with the `new_event()` method.
    This creates an event within the process container, and provides
    methods for writing particle data to the event.
    Repeatedly opening this context writes successive events under
    the same process.

    Parameters
    ----------
    path : str
        Filepath for output.
    """
    def __init__(self, path: str):
        self.path = path

    def __enter__(self):
        self._buffer = h5py.File(self.path, 'w', libver='latest')
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._buffer.close()

    def new_process(self, name: str) -> _ProcessWriter:
        """Returns a context handler object for storage of data in
        a given hard process.

        Events can be iteratively added to this process by repeatedly
        calling the `new_event()` method, which itself returns a context
        handler object.

        Parameters
        ----------
        key : str
            Arbitrary name, used to look up data with reader.

        """
        return _ProcessWriter(self, key=name)
