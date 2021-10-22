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

PMU_DTYPE = [
        ("x", TYPE['float']),
        ("y", TYPE['float']),
        ("z", TYPE['float']),
        ("e", TYPE['float'])
        ]

EDGE_DTYPE = [
        ("in", TYPE['int']),
        ("out", TYPE['int']),
        ]
