from functools import partial
from fractions import Fraction
import os

import numpy as np


CURRENT_DIR = str(os.path.dirname(os.path.realpath(__file__)))

def frac(num_str, obj_mode=False):
    """Converts string formatted fraction into number.

    Keyword arguments:
        num_str (str) -- string rep of rational number, eg. '1/2'
        obj_mod (bool) -- if True returns a fractions.Fraction object,
            if False returns a float

    example:
        In [1]: frac('1/2')
        Out [1]: 0.5

    Note on missing data:
        if passed empty string, will return 0,
        if passed '?', will return NaN
        other edge cases will raise a ValueError

    """
    if obj_mode == False:
        cast_frac = lambda inp: float(Fraction(inp))
    elif obj_mode == True:
        cast_frac = Fraction
    try:
        return cast_frac(num_str)
    except ValueError:
        if num_str == '':
            return cast_frac('0/1')
        elif num_str == '?':
            return np.nan

class LookupPDG:
    import pandas as __pd

    def __init__(self, frac_obj=False):
        frac_cols = ['I', 'G', 'P', 'C', 'Charge']
        cast_frac = partial(frac, obj_mode=frac_obj)
        converters = dict.fromkeys(frac_cols, cast_frac)
        lookup_table = self.__pd.read_csv(
                CURRENT_DIR + '/_pdg.csv', sep=',', comment='#',
                converters=converters)
        lookup_table.columns = lookup_table.columns.str.lower()
        self.__lookup = lookup_table.set_index('id')

    def pdg_properties(self, pdgs: np.ndarray, props: list) -> np.recarray:
        """Returns the physical properties of a sequence of particles
        based on their pdg code.

        Parameters
        ----------
        pdgs : iterable of integers
            The pdg codes of the particles to query.
        props : iterable of strings
            The properties you wish to obtain for the particles.
            Valid options are:
                - name
                - charge
                - mass
                - massupper
                - masslower
                - quarks
                - width
                - widthupper
                - widthlower
                - i (isospin)
                - g (G-parity)
                - p (space-parity)
                - c (charge-parity)
                - latex
        
        Returns
        -------
        pdg_properties : numpy record array
            Record array containing requested data for each particle in
            the order given by input pdgs, subscriptable by string field
            names or using object oriented dot notation.

        Examples
        --------
        >>> from heparchy.read import HepReader

        >>> with HepReader('showers.hdf5') as f:
        ...     with f.read_process(name='top') as process:
        ...         event = process.read_event(0)
        ...             pdg = event.pdg
        >>> pdg
        array([2212,   21,   21, ...,   22,   22,   22])

        >>> from heparchy.pdg import LookupPDG
        """
        props = list(props)
        pdg_ids, pdg_inv_idxs = np.unique(pdgs, return_inverse=True)
        uniq_data = self.__lookup.loc[pdg_ids][props]
        uniq_data = uniq_data.to_records()
        data = uniq_data[pdg_inv_idxs]
        return data
