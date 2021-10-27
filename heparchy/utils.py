import numpy as np

from heparchy import TYPE, PMU_DTYPE, EDGE_DTYPE, REAL_TYPE


def structure_pmu(array: np.ndarray) -> np.ndarray:
    """Helper function to convert 4 column array into structured array
    representing 4-momenta of particles.
    
    Parameters
    ----------
    array : numpy ndarray of floats, with shape (num particles, 4)
        The 4-momenta of the particles, arranged in columns.
        Columns must be in order (x, y, z, e).

    See also
    --------
    structure_pmu_components : structured array from seperate 1d arrays
                               of momentum components.

    Notes
    -----
    As the data-type of the input needs to be recast, the output is
    a copy of the original data, not a view on it. Therefore it uses
    additional memory, so later changes to the original will not
    affect the returned array, and vice versa.
    """
    if array.dtype != PMU_DTYPE:
        struc_array = array.astype(REAL_TYPE)
        struc_array =  struc_array.view(dtype=PMU_DTYPE, type=np.ndarray)
        struc_pmu = struc_array.copy().squeeze()
    else:
        struc_pmu = array
    return struc_pmu

def structure_pmu_components(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                             e: np.ndarray) -> np.ndarray:
    """Helper function to convert 1d arrays of 4-momentum components
    for particles into a single structured array.

    Parameters
    ----------
    x, y, z, e : numpy ndarray of floats
        components of the 4-momenta

    See also
    --------
    structure_pmu : structured array from single array with 4 columns.
    """
    data = list(x, y, z, e)
    shapes = set(col.shape for col in data)
    if len(shapes) > 1:
        raise ValueError('Shapes of all input arrays must be equal')
    return structure_pmu(np.vstack(data).T)

def unstructure_pmu(struc_array: np.ndarray,
                         dtype=REAL_TYPE) -> np.ndarray:
    """Returns view of 4-momentum data in columns (x, y, z, e) without
    as ordinary ndarray, without named columns.

    Parameters
    ----------
    struc_array : numpy ndarray with structured dtype names (x, y, z, e)
        Input structured 4-momentum array.
    dtype : data-type, optional
        Data type of named columns in array. Important if you created
        the structured array manually, rather than relying on the
        heparchy helper functions.
        Omitting results in default of '<f4' or np.int32.

    Notes
    -----
    As this returns a view object on the original data, modifying
    elements in the resulting array will also modify the original.
    """
    return struc_array.view(dtype=dtype).reshape((-1, 4))

def structure_edges(edges: np.ndarray) -> np.ndarray:
    if edges.dtype != EDGE_DTYPE:
        struc_array = edges.astype(TYPE['int'])
        struc_array =  struc_array.view(dtype=EDGE_DTYPE, type=np.ndarray)
        struc_edges = struc_array.copy().squeeze()
    else:
        struc_edges = edges
    return struc_edges

def unstructure_edges(struc_array: np.ndarray,
                         dtype=TYPE['int']) -> np.ndarray:
    return struc_array.view(dtype=dtype).reshape((-1, 2))
