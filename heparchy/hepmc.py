from functools import partial
from itertools import chain

import numpy as np


class HepMC:
    import pyhepmc_ng as __hepmc
    import networkx as __nx
    import vector as __vector


    def __init__(self, path, signal_vertices=None):
        self.path = path
        self._leaf_id = lambda pcl: -abs(pcl.id)
        self.__unpack = lambda nest: tuple(chain.from_iterable(nest))
        self.__num_signals = len(signal_vertices)
        self.__vtx_to_graph = partial(
                self.__vtx_to_graph,
                signal_vertices=signal_vertices
                )
        self.__pcl_to_leaf = partial(self.__pcl_to_edge, leaf=True)
        self.__empty_int = -2_000_000_000 # a sentinel for empty int data

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
        self.__content = self.__buffer.read()
        (self.__edges,
         self.__pmu,
         self.__pdg,
         self.__is_leaf) = self.__pcl_data(self.__content.particles)
        graph = self.graph()
        self.__signal_vtxs = graph['signal_vertices']
        self.__graph = self.__to_networkx(graph['edges'], graph['nodes'])
        return self

    def __pcl_data(self, pcls):
        """Returns 4-momenta of final state particles.
        """
        vtx_id = lambda vtx: int(vtx.id) if vtx != None else self.__empty_int
        def pcl_data(pcl):
            edge_idxs = (vtx_id(pcl.production_vertex),
                         vtx_id(pcl.end_vertex))
            pmu, pdg, status = tuple(pcl.momentum), pcl.pid, pcl.status
            return edge_idxs, pmu, pdg, status
        edges, pmus, pdgs, statuses = zip(*map(pcl_data, pcls))
        edges = np.fromiter(edges, dtype='<i4')
        pmu = self.__vector.array(
                [pmus],
                dtype=[("x", '<f'), ("y", '<f'), ("z", '<f'), ("e", '<f')]
                )
        pdg = np.fromiter(pdgs, dtype='<i4')
        is_leaf = np.fromiter(
                map(lambda status: status == 1, statuses), dtype='<?')
        return edge, pmu, pdg, is_leaf

    def __pcl_to_edge(self, pcl, leaf=False):
        """Takes a pyhepmc_ng particle object and outputs a tuple
        representation of a directed edge, connecting production and
        end vertices for the particle.
        """
        prod_vtx = pcl.production_vertex
        end_vtx = pcl.end_vertex
        data = {
            'pdg': pcl.pid,
            'leaf': leaf,
            'pmu': tuple(pcl.momentum)
            }
        if leaf: # final state pcl
            pcl_id = self._leaf_id(pcl)
            vtx_id = abs(prod_vtx.id)
            return (vtx_id, pcl_id, data)
        elif prod_vtx and end_vtx: # virtual pcl
            return (abs(prod_vtx.id), abs(end_vtx.id), data)

    def __vtx_to_graph(self, vtx, signal_vertices=None):
        """Takes a given vertex and outputs edges and nodes associated.
        Keyword arguments:
            vtx (pyhepmc_ng GenVertex): interaction vertex from hepmc
            signal (iterable): iterable containing pdgs of outgoing
                particles in signal production vertex
        """
        pcls_in = list(vtx.particles_in)
        pcls_out = list(vtx.particles_out)
        pcls_leaf = set(pcl for pcl in pcls_out if pcl.status == 1)
        num_leaves = len(pcls_leaf)
        edges = map(self.__pcl_to_edge, pcls_in)
        if num_leaves > 0:
            edges_leaf = map(self.__pcl_to_leaf, pcls_leaf)
        else:
            edges_leaf = tuple()
        edges = list(edges) + list(edges_leaf)
        nodes = [abs(vtx.id)]
        is_signal = (False)
        if signal_vertices is not None:
            is_signal = list(False for signal_vtx in signal_vertices)
            pdgs_in = set(pcl.pid for pcl in pcls_in)
            pdgs_out = set(pcl.pid for pcl in pcls_out)
            for idx, signal_vtx in enumerate(signal_vertices):
                signal_in = set(signal_vtx['in'])
                signal_out = set(signal_vtx['out'])
                if (signal_out.intersection(pdgs_out) == signal_out
                    and signal_in.intersection(pdgs_in) == signal_in
                    ):
                    is_signal[idx] = True
            is_signal = tuple(is_signal)
        return edges, nodes, is_signal

    def __to_networkx(self, edges, nodes):
        """Takes a GenEvent object from pyhepmc_ng and converts it to
        a networkx graph.
        """
        shower = self.__nx.DiGraph()
        shower.add_nodes_from(nodes)
        shower.add_edges_from(edges)
        return shower

    def graph(self):
        """Exposes a graph representation the current event, from the
        hard event to the parton shower.

        Returns a dictionary containing edges, nodes, and
        the node id for the signal vertex, if signal was provided to
        the object instantiation.

        Graph representation:
            Edges (tuple of tuples): each nested tuple represents an
                edge in the graph. Edges in the graph are equivalent
                to particles in the event.
                Tuples are composed of three entries:
                    1. ids of production vertices
                    2. ids of end vertices
                    3. particle data dictionary
                        - pdg (int): pdg particle id code
                        - leaf (bool): True if final state
                        - pmu (tuple): 4-momenta (e, px, py, pz)
            Nodes (tuple of ints): entries contain the id handle
                of each node in the graph
        """
        graph_map = map(self.__vtx_to_graph, self.__content.vertices)
        edges, nodes, is_signal = zip(*graph_map)
        edges = self.__unpack(edges)
        nodes = self.__unpack(nodes)
        is_signal = zip(*is_signal)
        signal_vtxs = []
        for signal_mask in is_signal:
            if True in signal_mask:
                signal_idx = signal_mask.index(True)
                signal_vtxs += [nodes[signal_idx]]
            else:
                signal_vtxs += [-1]
        signal_vtxs = tuple(signal_vtxs)
        return {'edges': edges, 'nodes': nodes, 'signal_vertices': signal_vtxs}

    def signal_descendants(self, signal_num, data=False):
        """Returns a dictionary containing sets of edges and node ids
        which descend from the signal vertex.
        """
        signal_vtx = self.__signal_vtxs[signal_num]
        if signal_vtx == -1:
            print("No signal provided or detected")
            edges, nodes = set(), set()
        else:
            nodes = self.__nx.descendants(self.__graph, signal_vtx)
            edges = self.__graph.edges(nbunch=nodes, data=data)
        return {'edges': edges, 'nodes': nodes}

    def leaf_edges(self, zipped=False):
        edges = self.__graph.edges
        is_leaf = edges.data('leaf')
        # filters out all edges where leaf = False:
        leaves = filter(lambda leaf: leaf[-1], is_leaf)
        in_vtx, out_vtx, _ = zip(*leaves)
        edge_idxs = zip(in_vtx, out_vtx)
        if not zipped:
            edge_idxs = set(edge_idxs)
        return edge_idxs

    def final_state_data(self, signal_num=-1):
        """Return the final state data for the event.

        Note: signal_num=-2 gives background.
        """
        if signal_num == -1:
            edges = self.__graph.edges(data=True)
        elif signal_num == -2:
            nodes_bg = set(self.__graph.nodes)
            for signal_iter in range(len(self.__signal_vtxs)):
                subgraph_signal = self.signal_descendants(
                        signal_num=signal_iter, data=True)
                nodes_fg = subgraph_signal['nodes']
                nodes_bg = nodes_bg.difference(nodes_fg)
            edges = self.__graph.edges(nbunch=nodes_bg, data=True)
        else:
            subgraph_signal = self.signal_descendants(
                    signal_num=signal_num, data=True)
            edges = subgraph_signal['edges']
        if len(edges) == 0:
            print(f'Signal number {signal_num} not present in event')
            return None
        # filters out all edges where leaf = False:
        leaves = filter(lambda edge: edge[-1]['leaf'], edges)
        leaves = tuple(leaves)
        # format data into numpy arrays:
        # momenta
        pmu = map(lambda edge: edge[-1]['pmu'], leaves)
        pmu = np.fromiter(
                chain.from_iterable(pmu),
                dtype='<f',
                )
        pmu = pmu.reshape((-1, 4))
        # pdg codes
        pdg = map(lambda edge: edge[-1]['pdg'], leaves)
        pdg = np.fromiter(pdg, dtype='<i4')

        in_vtx, out_vtx, _ = zip(*leaves)
        edge_idxs = zip(in_vtx, out_vtx)
        return {
                'edges': tuple(edge_idxs),
                'pmu': pmu,
                'pdg': pdg
                }
