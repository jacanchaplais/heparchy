import math
import warnings
from functools import wraps
from typing import Any, Callable


def event_key_format(evt_num: int, evts_per_chunk: int) -> str:
    evt_idx = evt_num % evts_per_chunk
    pad_len = math.ceil(math.log10(evts_per_chunk))
    return f"evt-{evt_idx:0{pad_len}}"


def chunk_key_format(chunk_num: int) -> str:
    return f"evt-set-{chunk_num:06}"


def deprecated(func: Callable[..., Any]) -> Callable[..., Any]:
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    @wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn(
            f"Call to deprecated function {func.__name__}.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return new_func
