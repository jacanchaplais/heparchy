import warnings
from itertools import chain

import numpy as np

from heparchy import TYPE, PMU_DTYPE
from heparchy.data import EventDataset
from heparchy.utils import structure_pmu

class HepMC:
    import pyhepmc_ng as __hepmc
    import networkx as __nx


    def __init__(self, path, signal_vertices=None):
        self.path = path
        self.data = EventDataset(
                edges = np.array([[0, 1]], dtype=TYPE['int']),
                pmu=np.array([[0.0, 0.0, 0.0, 0.0]], dtype=TYPE['float']),
                pdg=np.array([1], dtype=TYPE['int']),
                final=np.array([False], dtype=TYPE['bool'])
                )
        # self.__signal_dicts = signal_vertices
        # if signal_vertices is not None:
        #     self.__signal_empty_msgs = [
        #             "At least one event does not contain signal vertex {"
        #             + f"in: {vtx['in']}, out: {vtx['out']}"
        #             + "}."
        #             for vtx in self.__signal_dicts
        #             ]
        #     for msg in self.__signal_empty_msgs:
        #         warnings.filterwarnings('once', message=msg)

    # context manager
    def __enter__(self):
        self.__buffer = self.__hepmc.open(self.path, 'r')
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__buffer.close()

    # iterable
    def __iter__(self):
        self.__iter = iter(self.__buffer)
        return self

    def __next__(self):
        # make contents of file available everywhere
        self.__content = next(self.__iter)
        # read in the particle data
        self.data.flush_cache()
        (self.data.edges,
         self.data.pmu,
         self.data.pdg,
         self.data.final) = self.__pcl_data()
        return self

    def __pcl_data(self):
        pcls = self.__content.particles
        node_id = lambda obj: int(obj.id)
        def pcl_data(pcl):
            edge_idxs = [pcl.production_vertex, pcl.end_vertex]
            edge_idxs = tuple(node_id(vtx) if vtx != None
                              else node_id(pcl)
                              for vtx in edge_idxs)
            pmu, pdg, status = tuple(pcl.momentum), pcl.pid, pcl.status
            return edge_idxs, pmu, pdg, status
        edges, pmus, pdgs, statuses = zip(*map(pcl_data, pcls))
        edges = np.fromiter(chain.from_iterable(edges), dtype=TYPE['int'])
        edges = edges.reshape((-1, 2))
        pmu = np.array(list(pmus), dtype=TYPE['float'])
        pmu = structure_pmu(pmu)
        pdg = np.fromiter(pdgs, dtype=TYPE['int'])
        is_leaf = np.fromiter(
                map(lambda status: status == 1, statuses), dtype=TYPE['bool'])
        return edges, pmu, pdg, is_leaf

    # @property
    # def neutrino_filter(self):
    #     abs_pdg = np.abs(self.__pdg) # treat matter and antimatter as the same
    #     nu_pdg = (12, 14, 16) # e, mu, tau
    #     return np.bitwise_and.reduce([abs_pdg != nu for nu in nu_pdg])

    # def eta_filter(self, abs_limit=2.5):
    #     return np.abs(self.__pmu.eta) < abs_limit

    # def pt_filter(self, low=0.5, high=np.inf):
    #     pt = self.__pmu.pt
    #     return np.bitwise_and(pt >= low, pt <= high)

