import tempfile
import gzip
import shutil
import warnings
from itertools import chain

import numpy as np

from heparchy.data.event import ShowerData
from heparchy.utils import structure_pmu


class HepMC:
    """Returns an iterator over events in the given HepMC file.
    Event data is provided as a `heparchy.data.ShowerData` object.

    Parameters
    ----------
    path : string
        Location of the HepMC file.
        If this file is compressed with gzip, a temporary decompressed
        file will be created and cleaned up within the scope of the
        context manager.

    See also
    --------
    `heparchy.data.ShowerData` : Container for Monte-Carlo shower data.

    Examples
    --------
    >>> with HepMC('test.hepmc.gz') as hep_f:
    ...     for event in hep_f:
    ...         print(event.pmu)
    [( 0.        ,  0.        ,  6.49999993e+03, 6.50000000e+03)
     ( 1.42635177, -1.2172366 ,  1.36418624e+03, 1.36418753e+03)
     ( 0.21473153, -0.31874408,  1.04370539e+01, 1.04441276e+01) ...
     (-2.40079685,  1.56211274,  2.78633756e+00, 3.99596030e+00)
     (-0.34612959,  0.37377605,  5.25064994e-01, 7.31578753e-01)
     (-0.00765114,  0.00780012, -8.58527843e-03, 1.38956379e-02)]
    [( ...

    Notes
    -----
    If you wish to keep a given event in memory, you must use
    `ShowerData`'s `copy()` method.
    This is because the `HepMC` iterator avoids the substantial cost of
    repeated object instantiation by re-using a single `ShowerData`
    instance, with its efficient setter methods to update the data it
    contains.
    """

    import pyhepmc_ng as __hepmc
    import networkx as __nx


    def __init__(self, path):
        from typicle import Types
        self.__types = Types()
        self.path = path
        self.__gunz_f = None
        self.data = ShowerData.empty()
    # context manager
    def __enter__(self):
        try:
            self.__buffer = self.__hepmc.open(self.path, 'r')
        except UnicodeDecodeError:
            self.__gunz_f = tempfile.NamedTemporaryFile()
            with gzip.open(self.path, 'rb') as gz_f:
                shutil.copyfileobj(gz_f, self.__gunz_f)
            self.__buffer = self.__hepmc.open(self.__gunz_f.name, 'r')
        except:
            raise
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__buffer.close()
        if self.__gunz_f is not None:
            self.__gunz_f.close()

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
        return self.data

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
        edges = np.fromiter(chain.from_iterable(edges), dtype=self.__types.int)
        edges = edges.reshape((-1, 2))
        pmu = np.array(list(pmus), dtype=self.__types.pmu[0][1])
        pmu = structure_pmu(pmu)
        pdg = np.fromiter(pdgs, dtype=self.__types.int)
        is_leaf = np.fromiter(
                map(lambda status: status == 1, statuses),
                dtype=self.__types.bool
                )
        return edges, pmu, pdg, is_leaf
