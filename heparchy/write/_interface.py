from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class StorageBase(ABC):
    @abstractmethod
    def new_process(self, name: str) -> ProcessBase:
        pass

class ProcessBase(ABC):
    @abstractmethod
    def set_string(self, proc_str: str) -> None:
        pass

    @abstractmethod
    def set_decay(self, in_pcls: tuple, out_pcls: tuple) -> None:
        pass

    @abstractmethod
    def set_com_energy(self, energy: float, unit: str) -> None:
        pass

    @abstractmethod
    def set_custom_meta(self, name: str, metadata: Any) -> None:
        pass

    @abstractmethod
    def new_event(self) -> EventBase:
        pass

class EventBase(ABC):
    @abstractmethod
    def __set_num_pcls(self, data: np.ndarray) -> None:
        pass

    @abstractmethod
    def set_edges(self, data: np.ndarray) -> None:
        pass

    @abstractmethod
    def set_pmu(self, data: np.ndarray) -> None:
        pass

    @abstractmethod
    def set_pdg(self, data: np.ndarray) -> None:
        pass

    @abstractmethod
    def set_mask(self, name: str, data: np.ndarray) -> None:
        pass

    @abstractmethod
    def set_custom_dataset(self, name: str, data: np.ndarray,
                           dtype: Any) -> None:
        pass
