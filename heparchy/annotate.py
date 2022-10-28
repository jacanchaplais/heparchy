from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt


IntVector = npt.NDArray[np.int32]
HalfIntVector = npt.NDArray[np.int16]
DoubleVector = npt.NDArray[np.float64]
BoolVector = npt.NDArray[np.bool_]
AnyVector = npt.NDArray[Any]  # until numpy sorts out structured typing
DataType = TypeVar("DataType")
Self = TypeVar("Self")
