from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class EventReaderBase(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Getter for event name"""

    @property
    @abstractmethod
    def count(self) -> int:
        """Getter for number of particles in event"""

    @property
    @abstractmethod
    def edges(self) -> np.ndarray:
        """Getter for edges in event generation DAG."""

    @property
    @abstractmethod
    def pmu(self) -> np.ndarray:
        """Getter for 4-momentum of all particles in event."""

    @property
    @abstractmethod
    def color(self) -> np.ndarray:
        """Getter for color pairs of all particles in event."""

    @property
    @abstractmethod
    def pdg(self) -> np.ndarray:
        """Getter for pdg codes of all particles in event."""

    @property
    @abstractmethod
    def final(self) -> np.ndarray:
        """Getter for mask identifying final state particles."""

    @property
    @abstractmethod
    def available(self) -> list:
        """Provides list of all dataset names in event."""

    @abstractmethod
    def mask(self, name: str) -> np.ndarray:
        """Getter for a mask of a given name over all particles in event."""

    @abstractmethod
    def get_custom(self, name: str):
        """Getter for user-defined dataset stored in event."""

    @abstractmethod
    def copy(self):
        """Returns a deepcopy of this dataclass instance."""


class ProcessReaderBase(ABC):
    """An iterator of EventReaderBase dataclass instances over the
    events nested in the process.
    """

    @property
    @abstractmethod
    def string(self) -> str:
        """Getter for the MadGraph style process string."""

    @property
    @abstractmethod
    def decay(self) -> dict:
        """Returns dictionary with two entries, describing the hard
        interaction for this process.

        Dictionary items
        ----------------
        in_pcls : np.ndarray
            The pdg codes of the incoming particles.
        out_pcls : np.ndarray
            The pdg codes of the outgoing particles.
        """

    @property
    @abstractmethod
    def com_energy(self) -> dict:
        """Returns dictionary with two entries, describing the
        centre-of-mass energy for this hard process.

        Dictionary items
        ----------------
        energy : float
            The value of the centre-of-mass energy.
        unit : string
            The unit of energy, eg. GeV.
        """

    @abstractmethod
    def get_custom_meta(self, name: str):
        """Returns user-defined piece of metadata."""
    
    @abstractmethod
    def __getitem__(self, evt_num):
        """Provides option to get EventReaderBase object without
        iteration. Instead user supplies event number.
        """


class ReaderBase(ABC):
    """Context manager for opening heparchy formatted files."""

    @abstractmethod
    def read_process(self, name: str) -> ProcessReaderBase:
        """Returns process reader iterator object."""
