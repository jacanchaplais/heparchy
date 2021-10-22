from copy import deepcopy
from functools import wraps

import attr
import numpy as np

from heparchy.utils import structure_edges
from heparchy import TYPE


def val_elems_int(instance, attribute, value):
    all_ints = set(map(type, value)) == {int} # True if elems all ints
    if not all_ints:
        raise TypeError("Inputs must be iterables of integers.")

def val_int_array(instance, attribute, value):
    pass

@attr.s(kw_only=True, frozen=True)
class SignalVertex:
    """Data structure to define which of the vertices represent a
    signal in an event, and which of the given vertex's descendants
    are to be followed in the subsequent showering.

    Each of the inputs are to be formatted as an iterable of integers,
    representing the pdgs of the particles.

    Keyword arguments:
        incoming: particles incident on the vertex
        outgoing: particles outbound from the vertex
        follow: outbound particles marked to have their children tracked
    """

    from typing import Set as __Set

    __PdgSet = __Set[int]
    __pdg_kwargs = dict(converter=set, validator=[val_elems_int])
    incoming: __PdgSet = attr.ib(**__pdg_kwargs)
    outgoing: __PdgSet = attr.ib(**__pdg_kwargs)
    follow: __PdgSet = attr.ib(**__pdg_kwargs)

@attr.s(on_setattr=attr.setters.convert)
class EventDataset:
    import pandas as __pd
    import networkx as __nx

    __array_kwargs = dict(
            eq=attr.cmp_using(eq=np.array_equal),
            )
    edges: np.ndarray = attr.ib(
            converter=structure_edges,
            **__array_kwargs)
    pmu: np.ndarray = attr.ib(
            **__array_kwargs)
    pdg: np.ndarray = attr.ib(
            **__array_kwargs)
    final: np.ndarray = attr.ib(
            **__array_kwargs)

    def flush_cache(self):
        try:
            self.__shower
        except AttributeError:
            pass
        else:
            del self.__shower

    def to_networkx(self, data=['pdg']):
        # form edges with pdg data on for easier ancestry tracking
        names = data
        data_rows = (getattr(self, name) for name in names)
        data_rows = zip(*data_rows)
        edge_dicts = (dict(zip(names, row)) for row in data_rows)
        edges = zip(self.edges['in'], self.edges['out'], edge_dicts)
        shower = self.__nx.DiGraph()
        shower.add_edges_from(edges)
        return shower

    def __requires_shower(func):
        @wraps(func)
        def inner(self, *args, **kwargs):
            try:
                self.__shower
            except AttributeError:
                self.__shower = self.to_networkx()
            return func(self, *args, **kwargs)
        inner.__doc__ = func.__doc__
        return inner

    def to_pandas(self, data=('pdg', 'final')):
        """Return event data in a single pandas dataframe.

        Parameters
        ----------
        data : iterable of strings or True
            The particle properties to include as columns.
            Options include edges, pdg, pmu, final (boolean mask).
            Can also specify edge_in or edge_out separately, and
            individual components of momenta, ie. x, y, z, or e.
            If set to True instead of an iterable, all data is included.
            Default: ('pdg', 'final')

        Notes
        -----
        This is particularly useful if you want to mask the data
        in complex ways involving multiple particle properties at once,
        using the dataframe's `query` method.
        """
        # collect column data into a dict, ready to pass to pandas
        cols = {
            'pdg': self.pdg,
            'final': self.final,
            'edge_in': self.edges['in'],
            'edge_out': self.edges['out'],
            }
        pmu_cols = list(self.pmu.dtype.names)
        cols.update({col_name: self.pmu[col_name] for col_name in pmu_cols})
        # check which data to include
        if data == True: # all of it
            return self.__pd.DataFrame(cols)
        else: # restricted selection
            try:
                iterator = iter(data)
            except TypeError:
                print("data must either be an iterable, or True")
            else:
                # define valid input
                col_names_edge = ['edge_in', 'edge_out']
                valid_col_names = list(cols.keys())
                valid_col_names = set(valid_col_names + col_names_edge
                                      + ['pmu', 'edges'])
                # validate user input
                user_col_names = set(data)
                invalid_col_names = user_col_names.difference(valid_col_names)
                # discard invalid input
                col_names = user_col_names.intersection(valid_col_names)
                col_names = list(col_names)
                if invalid_col_names: # let user know of invalid input
                    print(f'Warning: {len(invalid_col_names)} invalid column '
                          + f'name(s) provided, {invalid_col_names}. '
                          + 'Omitting.')
                if 'edges' in col_names: # convert abbreviations
                    col_names.remove('edges')
                    col_names += col_names_edge
                if 'pmu' in col_names:
                    col_names.remove('pmu')
                    col_names += pmu_cols
                return self.__pd.DataFrame( # populate dataframe with selection
                        {key: cols[key] for key in col_names})

    @__requires_shower
    def vertex_pdg(self):
        """Returns a generator object which loops over all interaction
        vertices in the event, yielding the vertex id, and pdg codes
        of incoming and outgoing particles to the vertex, respectively.

        Examples
        --------

        """
        shower = self.__shower
        for vertex in shower:
            incoming = shower.in_edges(vertex, data='pdg')
            outgoing = shower.out_edges(vertex, data='pdg')
            if incoming and outgoing:
                vtxs_in, _, pdgs_in = zip(*incoming)
                _, vtxs_out, pdgs_out = zip(*outgoing)
                yield vertex, set(pdgs_in), set(pdgs_out)
    
    @__requires_shower
    def signal_mask(self, signal_vertices):
        is_signal = []
        for vertex, pdgs_in, pdgs_out in self.vertex_pdg():
            is_signal.append(tuple(
                [vertex] + [sig_vtx.incoming.issubset(pdgs_in)
                            and sig_vtx.outgoing.issubset(pdgs_out)
                            for sig_vtx in signal_vertices]
                ))
        is_signal = list(zip(*is_signal))
        vertices = is_signal[0]
        is_signal = is_signal[1:]
        signal_ids = [vertices[sig_list.index(True)] for sig_list in is_signal]
        signal_masks = []
        for signal_id, signal_vertex in zip(signal_ids, signal_vertices):
            follow_masks = dict()
            edges_out_info = self.__shower.out_edges(signal_id, data='pdg')
            follow_edges = filter(
                    lambda edge: edge[-1] in signal_vertex.follow,
                    edges_out_info
                    )
            _, follow_ids, follow_pdgs = zip(*follow_edges)
            for follow_id, follow_pdg in zip(follow_ids, follow_pdgs):
                desc_vtxs = self.__nx.descendants(self.__shower, follow_id)
                desc_edges = list(self.__shower.edges(nbunch=desc_vtxs))
                desc_edges = structure_edges(np.array(desc_edges))
                mask = np.isin(self.edges, desc_edges, assume_unique=True)
                follow_masks.update({follow_pdg: mask})
            signal_masks += [follow_masks]
        return signal_masks

    def copy(self):
        return deepcopy(self)
