"""
`heparchy.write.hdf`
===================

Provides the interface to write HEP data to the heparchy HDF5 format.
"""
from __future__ import annotations
from typing import (
        Any, Optional, Union, Sequence, Dict, Set, Iterator, Generic, TypeVar,
        Callable, Tuple)
from collections.abc import MutableMapping, Iterable
from pathlib import Path
from enum import Enum
import warnings

import numpy.typing as npt
import h5py
from h5py import Group, File

import heparchy
from heparchy.utils import deprecated, event_key_format
from .base import WriterBase, ProcessWriterBase, EventWriterBase
from heparchy.annotate import (
        IntVector, HalfIntVector, DoubleVector, BoolVector, AnyVector)


WRITE_ONLY_MSG = "Attribute is write-only."
MapGeneric = TypeVar("MapGeneric")
IterItem = TypeVar("IterItem")
WriterType = Union["HdfEventWriter", "HdfProcessWriter"]


class OverwriteWarning(RuntimeWarning):
    """A warning to be raised when a user writes a piece of data twice."""


class Compression(Enum):
    """Sets the compression algorithm used to store the datasets."""
    LZF = "lzf"
    GZIP = "gzip"


def _mask_setter(writer: HdfEventWriter, name: str, data: BoolVector) -> None:
    writer._set_num_pcls(data)
    writer._mk_dset(
            name=name,
            data=data,
            shape=(writer.num_pcls,),
            dtype=writer._types.bool,
            )


def _custom_setter(writer: HdfEventWriter, name: str, data: AnyVector) -> None:
    writer._mk_dset(
            name=name,
            data=data,
            shape=data.shape,
            dtype=data.dtype,
            )


def _meta_setter(writer: Union[HdfProcessWriter, HdfEventWriter], name: str,
                 data: Any) -> None:
    writer._grp.attrs[name] = data


class MapWriter(Generic[MapGeneric], MutableMapping[str, MapGeneric]):
    """Provides a dictionary-like interface for writing user-named
    datasets or metadata.
    """
    def __init__(self,
                 writer: WriterType,
                 setter_func: Callable[[WriterType, str, MapGeneric], None]
                 ) -> None:
        self.writer = writer
        self._names: Set[str] = set()
        self._setter_func = setter_func

    def __repr__(self) -> str:
        dset_repr = "<Write-Only Data>"
        kv = ", ".join(map(lambda k: f"\'{k}\': {dset_repr}", self._names))
        return "{" + f"{kv}" + "}"

    def __len__(self) -> int:
        return len(self._names)

    def __getitem__(self, name: str) -> MapGeneric:
        raise AttributeError(WRITE_ONLY_MSG)

    def __setitem__(self, name: str, data: MapGeneric) -> None:
        self._setter_func(self.writer, name, data)
        self._names.add(name)

    def __delitem__(self, name: str) -> None:
        self._names.remove(name)
        if self._setter_func.__name__ == "_meta_setter":
            del self.writer._grp.attrs[name]
            return
        del self.writer._grp[name]

    def __iter__(self) -> Iterator[str]:
        for name in self._names:
            yield name

    def _flush(self) -> Tuple[str, ...]:
        data = tuple(self._names)
        self._names = set()
        return data


class HdfEventWriter(EventWriterBase):
    """Context manager interface to create and write events.

    Attributes
    ----------
    """
    __slots__ = ("_types", "__grp_obj", "_idx", "num_pcls", "_evt", "masks",
                 "custom", "custom_meta", "_num_edges")

    def __init__(self, proc: HdfProcessWriter) -> None:
        from typicle import Types
        self._types = Types()
        self._proc = proc  # pointer to parent group obj
        self._idx = proc._evt_idx  # index for current event
        self.num_pcls: Optional[int] = None
        self._grp: Group
        self.masks: MapWriter[BoolVector]
        self.custom: MapWriter[AnyVector]
        self.custom_meta: MapWriter[Any]
        self._num_edges = 0

    def __enter__(self: HdfEventWriter) -> HdfEventWriter:
        self._grp = self._proc._grp.create_group(
                event_key_format(self._idx))
        self.masks = MapWriter(self, _mask_setter)
        self.custom = MapWriter(self, _custom_setter)
        self.custom_meta = MapWriter(self, _meta_setter)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        # increasing the index for the group to create next event
        # also acts as running counter for number events
        if self.num_pcls is None:
            self.num_pcls = 0
        self._grp.attrs["num_pcls"] = self.num_pcls
        self._grp.attrs["mask_keys"] = self.masks._flush()
        self._grp.attrs["custom_keys"] = self.custom._flush()
        self._grp.attrs["custom_meta_keys"] = self.custom_meta._flush()
        self._proc._evt_idx += 1

    def _set_num_pcls(self, data: AnyVector) -> None:
        shape = data.shape
        num_pcls = shape[0]

        if self.num_pcls is None:
            self.num_pcls = num_pcls
        elif num_pcls != self.num_pcls:
            raise ValueError(
                    "Datasets within same event must have the same "
                    "number of particles (rows, if 2D array). "
                    f"Previous datasets had {self.num_pcls}, "
                    f"attempted dataset would have {num_pcls}."
                    )
        else:
            return

    def _mk_dset(
            self, name: str, data: AnyVector, shape: tuple,
            dtype: npt.DTypeLike, is_mask: bool = False) -> None:
        """Generic dataset creation and population function.
        Wrap in methods exposed to the user interface.
        """
        if name in self._grp:
            warnings.warn(f"Overwriting {name}", OverwriteWarning)
            del self._grp[name]
        # check data can be broadcasted to dataset:
        if data.squeeze().shape != shape:
            raise ValueError(
                    f"Input data shape {data.shape} "
                    f"incompatible with dataset shape {shape}."
                    )
        kwargs: Dict[str, Any] = dict(
                name=name,
                shape=shape,
                dtype=dtype,
                shuffle=True,
                compression=self._proc._file_obj._cmprs.value,
                )
        cmprs_lvl = self._proc._file_obj._cmprs_lvl
        if cmprs_lvl is not None:
            kwargs["compression_opts"] = cmprs_lvl
        dset = self._grp.create_dataset(**kwargs)
        dset[...] = data
        dset.attrs["mask"] = is_mask

    @property
    def edges(self) -> AnyVector:
        raise AttributeError(WRITE_ONLY_MSG)

    @edges.setter
    def edges(self, data: AnyVector) -> None:
        self._mk_dset(
                name="edges",
                data=data,
                shape=data.shape,
                dtype=self._types.edge,
                )
        self._num_edges = len(data)

    @property
    def edge_weights(self) -> DoubleVector:
        raise AttributeError(WRITE_ONLY_MSG)

    @edge_weights.setter
    def edge_weights(self, data: DoubleVector) -> None:
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
                dtype=self._types.double,
                )

    @deprecated
    def set_edges(
            self,
            data: AnyVector,
            weights: Optional[DoubleVector] = None,
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
            self._set_num_pcls(data)
        if weights is not None:
            num_edges = len(data)
            num_weights = len(weights)
            if num_edges != num_weights:
                raise ValueError(
                    f"Incompatible shapes. Number of edges = {num_edges} and "
                    + f"number of weights = {num_weights}. Must be same size."
                    )
            self._mk_dset(
                    name='edge_weights',
                    data=weights,
                    shape=weights.shape,
                    dtype=self._types.double,
                    )
        self._mk_dset(
                name='edges',
                data=data,
                shape=data.shape,
                dtype=self._types.edge,
                )

    @deprecated
    def set_pmu(self, data: AnyVector) -> None:
        self.pmu = data

    @property
    def pmu(self) -> AnyVector:
        """Write 4-momentum for all particles to event.

        Parameters
        ----------
        data : 2d array of floats.
            Each row contains momentum of one particle,
            in order [px, py, pz, e].
        """
        raise AttributeError(WRITE_ONLY_MSG)

    @pmu.setter
    def pmu(self, data: AnyVector) -> None:
        self._set_num_pcls(data)
        self._mk_dset(
                name='pmu',
                data=data,
                shape=data.shape,
                dtype=self._types.pmu,
                )

    @deprecated
    def set_color(self, data: AnyVector) -> None:
        self.color = data

    @property
    def color(self) -> AnyVector:
        """Write color / anticolor pairs for all particles to event.

        Parameters
        ----------
        data : 2d array of ints.
            Each row contains color / anticolor values respectively.
        """
        raise AttributeError(WRITE_ONLY_MSG)

    @color.setter
    def color(self, data: AnyVector) -> None:
        self._set_num_pcls(data)
        self._mk_dset(
                name='color',
                data=data,
                shape=data.shape,
                dtype=self._types.color,
                )

    @deprecated
    def set_pdg(self, data: IntVector) -> None:
        self.pdg = data

    @property
    def pdg(self) -> IntVector:
        """Pdg codes for all particles in event.

        Parameters
        ----------
        data : iterable or 1d numpy array
            Iterable of ints containing pdg codes for every
            particle in the event.
        """
        raise AttributeError(WRITE_ONLY_MSG)

    @pdg.setter
    def pdg(self, data: IntVector) -> None:
        self._set_num_pcls(data)
        self._mk_dset(
                name='pdg',
                data=data,
                shape=(self.num_pcls,),
                dtype=self._types.int,
                )

    @deprecated
    def set_status(self, data: IntVector) -> None:
        self.status = data

    @property
    def status(self) -> IntVector:
        raise AttributeError(WRITE_ONLY_MSG)

    @status.setter
    def status(self, data: IntVector) -> None:
        """Write status codes for all particles to event.
        
        Parameters
        ----------
        data : iterable or 1d numpy array
            Iterable of ints containing status codes for every
            particle in the event.
        """
        self._set_num_pcls(data)
        self._mk_dset(
                name='status',
                data=data,
                shape=(self.num_pcls,),
                dtype=self._types.int,
                )

    @deprecated
    def set_helicity(self, data: HalfIntVector) -> None:
        self.helicity = data

    @property
    def helicity(self) -> HalfIntVector:
        raise AttributeError(WRITE_ONLY_MSG)

    @helicity.setter
    def helicity(self, data: HalfIntVector) -> None:
        """Write helicity values for all particles to event.

        Parameters
        ----------
        data : iterable or 1d numpy array
            Iterable of floats containing helicity values for every
            particle in the event.
        """
        self._set_num_pcls(data)
        self._mk_dset(
                name='helicity',
                data=data,
                shape=(self.num_pcls,),
                dtype=self._types.helicity,
                )
    
    @deprecated
    def set_mask(self, name: str, data: BoolVector) -> None:
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
        self.masks[name] = data

    @deprecated
    def set_custom_dataset(
            self, name: str,
            data: AnyVector,
            dtype: npt.DTypeLike,
            strict_size: bool = True) -> None:
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
        if strict_size is True:
            self._set_num_pcls(data)
        self.custom[name] = data

    @deprecated
    def set_custom_meta(self, name: str, metadata: Any) -> None:
        """Store custom metadata to the event.

        Parameters
        ----------
        name : str
            Handle to access the metadata at read time.
        metadata : str, int, float, or iterables thereof
            The data you wish to store.
        """
        self.custom_meta[name] = metadata


class HdfProcessWriter(ProcessWriterBase):
    """Context manager interface to create and write processes.

    Attributes
    ----------
    """
    def __init__(self, file_obj: HdfWriter, key: str) -> None:
        self._file_obj = file_obj
        self.key = key
        self._evt_idx = 0
        self._grp: Group
        self.custom_meta: MapWriter[Any]

    def __enter__(self: HdfProcessWriter) -> HdfProcessWriter:
        self._grp = self._file_obj._buffer.create_group(self.key)
        self.custom_meta = MapWriter(self, _meta_setter)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        # count all of the events and write to attribute
        self._grp.attrs["num_evts"] = self._evt_idx

    @deprecated
    def set_string(self, proc_str: str) -> None:
        """Writes the string formatted underlying event to the
        process metadata.
        
        Parameters
        ----------
        proc_str : str
            MadGraph formatted string representing the hard event,
            eg. p p > t t~
        """
        self.process_string = proc_str

    @property
    def process_string(self) -> str:
        raise AttributeError(WRITE_ONLY_MSG)

    @process_string.setter
    def process_string(self, value: str) -> None:
        self._grp.attrs["process"] = value

    @deprecated
    def set_decay(self, in_pcls: Sequence[int], out_pcls: Sequence[int]
                  ) -> None:
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

    @property
    def signal_pdgs(self) -> IntVector:
        raise AttributeError(WRITE_ONLY_MSG)

    @signal_pdgs.setter
    def signal_pdgs(self, value: Sequence[int]) -> None:
        self._grp.attrs["signal_pdgs"] = value

    @deprecated
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

    @property
    def com_energy(self) -> Tuple[float, str]:
        raise AttributeError(WRITE_ONLY_MSG)

    @com_energy.setter
    def com_energy(self, value: Tuple[float, str]) -> None:
        self._grp.attrs["com_e"] = value[0]
        self._grp.attrs["e_unit"] = value[1]

    @deprecated
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

    def new_event(self) -> HdfEventWriter:
        return HdfEventWriter(self)

    def event_iter(self, iterable: Iterable[IterItem]
                   ) -> Iterator[Tuple[HdfEventWriter, IterItem]]:
        """Wraps an iteratable object, returning a new iterator which
        yields a new writeable event object followed by the value
        obtained from the passed iterator.

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
    path : Path, str
        Filepath for output.
    """
    def __init__(self,
                 path: Union[Path, str],
                 compression: Union[str, Compression] = Compression.GZIP,
                 compression_level: Optional[int] = 4,
                 ) -> None:
        self.path = Path(path)
        self._buffer: File
        if isinstance(compression, str):
            compression = Compression(compression.lower())
        self._cmprs = compression
        if compression is Compression.LZF:
            compression_level = None
        self._cmprs_lvl = compression_level

    def __enter__(self: HdfWriter) -> HdfWriter:
        self._buffer = h5py.File(self.path, 'w', libver='latest')
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self._buffer.attrs["layout"] = "heparchy"
        self._buffer.attrs["version_tuple"] = tuple(
                map(str, heparchy.__version_tuple__))
        self._buffer.attrs["version"] = heparchy.__version__
        self._buffer.close()

    def new_process(self, name: str) -> HdfProcessWriter:
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
        return HdfProcessWriter(self, key=name)
