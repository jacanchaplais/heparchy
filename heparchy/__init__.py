import numpy as np


TYPE = {
    'bool': '<?',
    'half': '<f2',
    'float': '<f4',
    'double': '<f8',
    'h_int': '<i2',
    'int': '<i4',
    'd_int': '<i8',
    }

# default type for floating point numbers set to double
REAL_TYPE = TYPE['double']

PMU_DTYPE = [
        ("x", REAL_TYPE),
        ("y", REAL_TYPE),
        ("z", REAL_TYPE),
        ("e", REAL_TYPE)
        ]

EDGE_DTYPE = [
        ("in", TYPE['int']),
        ("out", TYPE['int']),
        ]


event_key_format = lambda evt_num: f'event_{evt_num:09}'
