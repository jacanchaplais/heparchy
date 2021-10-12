from functools import partial
from itertools import chain

import numpy as np


class HepMC:
    import pyhepmc_ng as __hepmc
    import networkx as __nx
    import vector as __vector


    def __init__(self, path, signal_vertices=None):
        self.path = path
        self.__signal_vertices = signal_vertices
        self.__empty_int = 2_000_000_000 # a sentinel for empty int data

    # context manager
    def __enter__(self):
        self.__buffer = self.__hepmc.open(self.path, 'r')
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__buffer.close()

    # iterable
    def __iter__(self):
        return self

    def __next__(self):
        # make contents of file available everywhere
        self.__content = self.__buffer.read()
        # read in the particle data
        (self.__edges,
         self.__pmu,
         self.__pdg,
         self.__is_leaf) = self.__pcl_data()
        # read in the vertex data
        self.__nodes, is_signal = self.__vtx_data()
        # set the vertices on which user defined signal is found
        signal_vtxs = np.array([-1 for vtx in self.__signal_vertices])
        signal_idx = np.argwhere(is_signal)
        signal_vtxs[signal_idx[:, 1]] = self.__nodes[signal_idx[:, 0]]
        self.__signal_vtxs = signal_vtxs
        self.__graph = self.__to_networkx()
        return self

    def __pcl_data(self):
        """Returns 4-momenta of final state particles.
        """
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
        edges = np.fromiter(chain.from_iterable(edges), dtype='<i4')
        edges = edges.reshape((-1, 2))
        pmu = self.__vector.array(
                list(pmus),
                dtype=[("x", '<f'), ("y", '<f'), ("z", '<f'), ("e", '<f')]
                )
        pdg = np.fromiter(pdgs, dtype='<i4')
        is_leaf = np.fromiter(
                map(lambda status: status == 1, statuses), dtype='<?')
        return edges, pmu, pdg, is_leaf

    def __vtx_data(self):
        if self.__signal_vertices is None:
            num_signals = 1
            def vtx_data(vtx):
                nodes = [int(vtx.id)]
                return nodes, (False)
        else:
            num_signals = len(self.__signal_vertices)
            def vtx_data(vtx):
                nodes = [int(vtx.id)]
                is_signal = self.__check_node_signal(
                    vtx, self.__signal_vertices)
                return nodes, is_signal
        nodes, signal = zip(*map(vtx_data, self.__content.vertices))
        nodes = np.fromiter(chain.from_iterable(nodes), dtype='<i4')
        # creates a flattened boolean array holding all signals:
        signal = np.fromiter(chain.from_iterable(signal), dtype='<?')
        signal = np.squeeze( # flattens array if no signals given
            signal.reshape((-1, num_signals)) # creates a col for each signal
            )
        return nodes, signal

    def __check_node_signal(self, vtx, signal_vertices):
        pcls_in = tuple(vtx.particles_in)
        pcls_out = tuple(vtx.particles_out)
        # get pdgs of pcls for comparison
        pdgs_in = set(pcl.pid for pcl in pcls_in)
        pdgs_out = set(pcl.pid for pcl in pcls_out)
        def check_signal(signal_vtx):
            signal_in = set(signal_vtx['in'])
            signal_out = set(signal_vtx['out'])
            return ( # true if both in and out pdgs are found on a vertex
                signal_out.intersection(pdgs_out) == signal_out
                and signal_in.intersection(pdgs_in) == signal_in
                )
        is_signal = map(check_signal, signal_vertices)
        return tuple(is_signal)

    def __to_networkx(self):
        """Takes a GenEvent object from pyhepmc_ng and converts it to
        a networkx graph.
        """
        edges = tuple(map(tuple, self.__edges))
        shower = self.__nx.DiGraph()
        shower.add_nodes_from(self.__nodes)
        shower.add_edges_from(edges)
        return shower

    def __signal_descendants(self, signal_num):
        """Returns a dictionary containing sets of edges and node ids
        which descend from the signal vertex.
        """
        signal_vtx = self.__signal_vtxs[signal_num]
        if signal_vtx == -1:
            print("No signal provided or detected")
            edges, nodes = set(), set()
        else:
            nodes = self.__nx.descendants(self.__graph, signal_vtx)
            edges = self.__graph.edges(nbunch=nodes)
        return {'edges': edges, 'nodes': nodes}

    @property
    def data(self):
        return {
            'pmu': np.array(self.__pmu),
            'pdg': self.__pdg,
            }

    @property
    def graph(self):
        return {
            'edges': self.__edges,
            'nodes': self.__nodes
            }

    @property
    def final_state_mask(self):
        return self.__is_leaf

    def eta_mask(self, abs_limit=2.5):
        return np.abs(self.__pmu.eta) < abs_limit

    def pt_mask(self, low=0.5, high=np.inf):
        pt = self.__pmu.pt
        return np.bitwise_and(pt >= low, pt <= high)

    def signal_mask(self, signal_num):
        desc = self.__signal_descendants(signal_num)
        edges, nodes = desc['edges'], desc['nodes']
        masks = map(
            lambda edge: np.bitwise_and.reduce(self.__edges == edge, axis=1),
            edges
            )
        signal_mask = np.bitwise_or.reduce(np.array(tuple(masks)), axis=0)
        return signal_mask
