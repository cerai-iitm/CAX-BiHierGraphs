import torch
from torch_geometric.data import Data
import numpy as np
from itertools import count
from heapq import heappop, heappush
from queue import PriorityQueue


def get_edge_mask_dict(graph, device):
    """
    Create a dictionary mapping etypes to learnable edge masks

    Parameters:
        graph: PyTorch Geometric HeteroData object

    Returns:
        edge_mask_dict: dictionary
            key=etype, value=torch.nn.Parameter with size number of etype edges
    """

    edge_mask_dict = {}

    for etype in graph.edge_types:
        num_edges = graph[etype].edge_index.size(1)
        num_nodes = graph.edge_type_subgraph([etype]).num_nodes

        std = torch.nn.init.calculate_gain("relu") * np.sqrt(2.0 / (2 * num_nodes))
        edge_mask_dict[etype] = torch.nn.Parameter(
            torch.randn(num_edges, device=device) * std
        )

    return edge_mask_dict


def hetero_k_hop_subgraph(graph, center_nodes, k, device, edge_weights, num_neighbors):
    """
    Compute the k-hop subgraph of a heterogenous graph

    Parameters:
        graph: PyTorch Geometric HeteroData object

        center_nodes: Dict[str, Tensor]
            A dictionary mapping node types to the IDs of nodes from which to start the k-hop computation

        k: int

        edge_weights: dictionary
            key=etype, value=torch.nn.Parameter with size number of etype edges

        num_neighbors: int

    Returns:
        A PyTorch Geometric HeteroData object containing the k-hop subgraph,
        mapping from the original node IDs to the subgraph node IDs
    """

    # initialize node and edge (neighbor) masks
    node_mask = {
        ntype: torch.empty((graph[ntype].x.size(0)), dtype=torch.bool).to(device)
        for ntype in graph.node_types
    }
    edge_mask = {}
    for etype in graph.edge_types:
        if "self" not in etype[1]:
            edge_mask[etype] = torch.empty(
                (graph[etype].edge_index.size(1)), dtype=torch.bool
            ).to(device)

    # all zero- to k-hop neighbors will be stored in a list where the index indicates the number of hops from the center node(s)
    subset = []
    subset_zero_hop = {}
    for ntype in graph.node_types:
        if ntype in center_nodes:
            subset_zero_hop[ntype] = center_nodes[ntype]
        else:
            subset_zero_hop[ntype] = torch.tensor([], dtype=torch.int).to(device)
    subset.append(subset_zero_hop)
    del subset_zero_hop

    for _ in range(k):
        # set all masks to False
        for ntype in node_mask:
            node_mask[ntype].fill_(False)
        # set the mask value of the ones found in the previous hop to True
        for ntype in node_mask:
            node_mask[ntype][subset[-1][ntype]] = True
        # get the one-hop neighbors of the nodes found in the previous hop
        subset_hop = {}
        for etype in graph.edge_types:
            if "self" not in etype[1]:
                if not edge_weights:
                    torch.index_select(
                        node_mask[etype[0]],
                        0,
                        graph[etype].edge_index[0],
                        out=edge_mask[etype],
                    )

                    if etype[2] not in subset_hop:
                        subset_hop[etype[2]] = graph[etype].edge_index[1][
                            edge_mask[etype]
                        ]
                    else:
                        subset_hop[etype[2]] = torch.cat(
                            (
                                subset_hop[etype[2]],
                                graph[etype].edge_index[1][edge_mask[etype]],
                            ),
                            dim=0,
                        )

                else:  # pick the top-k edge-weighted nodes
                    for idx, node in enumerate(node_mask[etype[2]]):
                        if node.item():
                            indices = torch.where(graph[etype].edge_index[1] == idx)[0]
                            all_neighbors = graph[etype].edge_index[0][indices]
                            neighbor_weights = edge_weights[etype][
                                indices[indices < edge_weights[etype].size(0)]
                            ]
                            k_neighbors = None
                            if neighbor_weights.shape[0] >= num_neighbors:
                                topk_neighbor_indices = torch.topk(
                                    neighbor_weights, num_neighbors, dim=0
                                )[1].squeeze(0)
                                k_neighbors = all_neighbors[topk_neighbor_indices]
                            else:
                                k_neighbors = all_neighbors
                            if etype[0] not in subset_hop:
                                subset_hop[etype[0]] = k_neighbors
                            else:
                                subset_hop[etype[0]] = torch.cat(
                                    (subset_hop[etype[0]], k_neighbors), dim=0
                                )

        # create empty tensors for node types not present in subset_hop
        for ntype in graph.node_types:
            if ntype not in subset_hop:
                subset_hop[ntype] = torch.tensor([]).to(device)

        subset.append(subset_hop)
        del subset_hop

    # combine all the subsets
    subgraph_dict = {}
    for ntype in graph.node_types:
        nodes = [subset_hop[ntype] for subset_hop in subset]
        subgraph_dict[ntype] = torch.cat(nodes, dim=0).unique()
    del subset

    # compute a previous ID to current ID mapping
    mapping = {}
    for ntype in subgraph_dict:
        mapping_ntype = {}
        for idx, node in enumerate(subgraph_dict[ntype]):
            mapping_ntype[node.item()] = idx
        mapping[ntype] = mapping_ntype

    return graph.subgraph(subgraph_dict), mapping


def hetero_src_tgt_khop_in_subgraph(
    src_ntype,
    src_nid,
    tgt_ntype,
    tgt_nid,
    graph,
    k,
    device,
    edge_weights=None,
    num_neighbors=3,
):
    """
    Find the k-hop subgraph around the source node and target node in the graph
    The output will be the union of two subgraphs

    Parameters:
        src_ntype: string
            Source node type

        src_nid : int
            Source node id

        tgt_ntype: string
            Target node type

        tgt_nid : int
            Target node id

        graph : PyTorch Geometric HeteroData object

        k: int
            Number of hops

        edge_weights: dictionary
            key=etype, value=torch.nn.Parameter with size number of etype edges

        num_neighbors: int

    Returns:
        sghetero_src_nid: int
            ID of the source node in the subgraph

        sghetero_tgt_nid: int
            ID of the target node in the subgraph

        sghetero : PyTorch Geometric HeteroData object
            Union of two k-hop subgraphs

        sghetero_feat_nid: Tensor
            The original graph node IDs of subgraph nodes, for feature identification
    """

    # Extract the k-hop graph centered at the (src, tgt) pair
    src_nid = src_nid.item() if torch.is_tensor(src_nid) else src_nid
    tgt_nid = tgt_nid.item() if torch.is_tensor(tgt_nid) else tgt_nid

    if src_ntype == tgt_ntype:
        pred_dict = {src_ntype: torch.tensor([src_nid, tgt_nid]).to(device)}
        sghetero, mapping = hetero_k_hop_subgraph(
            graph, pred_dict, k, device, edge_weights, num_neighbors
        )
        sghetero_src_nid = mapping[src_ntype][src_nid]
        sghetero_tgt_nid = mapping[tgt_ntype][tgt_nid]
    else:
        pred_dict = {
            src_ntype: torch.tensor([src_nid]).to(device),
            tgt_ntype: torch.tensor([tgt_nid]).to(device),
        }
        sghetero, mapping = hetero_k_hop_subgraph(
            graph, pred_dict, k, device, edge_weights, num_neighbors
        )
        sghetero_src_nid = mapping[src_ntype][src_nid]
        sghetero_tgt_nid = mapping[tgt_ntype][tgt_nid]

    # sghetero_feat_nid = sghetero.ndata[dgl.NID]

    return sghetero_src_nid, sghetero_tgt_nid, sghetero, mapping  # , sghetero_feat_nid


def remove_edges_of_high_degree_nodes(graph, max_degree, always_preserve, device):
    """
    Remove the edges of all the nodes with a degree greater than max_degree, except the ones in always_preserve

    Parameters:
        graph: Union[PyTorch Geometric HeteroData, PyTorch Geometric Data]

        max_degree: int

        always_preserve: dictionary
            key=node_type, value=torch.tensor with the indices of the nodes to be preserved

    Returns:
        The pruned graph with a maximum possible degree of max_degree
    """

    try:
        if isinstance(graph, Data):
            # TODO: add support for homogenous graphs
            raise NotImplementedError

        # find the in-degree of every node
        in_degree = {
            ntype: torch.zeros(graph[ntype].x.size(0), dtype=torch.int).to(device)
            for ntype in graph.node_types
        }
        for edge_type in graph.edge_types:
            for tgt_node in graph[edge_type].edge_index[1]:
                in_degree[edge_type[2]][tgt_node] += 1

        # prune all the edges of high degree nodes except the edges of the nodes in always_preserve
        high_degree_mask = {}
        for ntype in graph.node_types:
            high_degree_mask[ntype] = in_degree[ntype] > max_degree
            high_degree_mask[ntype][always_preserve[ntype]] = False
        del in_degree

        for etype in graph.edge_types:
            src, dst = graph[etype].edge_index[0], graph[etype].edge_index[1]
            src_nodes_high_degree = torch.arange(
                graph[etype[0]].x.size(0), device=device
            )[high_degree_edge_mask[etype[0]]]
            dst_nodes_high_degree = torch.arange(
                graph[etype[2]].x.size(0), device=device
            )[high_degree_edge_mask[etype[2]]]
            high_degree_edge_mask = torch.isin(src, src_nodes_high_degree) | torch.isin(
                dst, dst_nodes_high_degree
            )
            graph[etype].edge_index = graph[etype].edge_index[:, high_degree_edge_mask]

        return graph

    except NotImplementedError as e:
        print(f"Support for homogenous graphs has not been added yet\n{e}")


def remove_edges_except_k_core_graph(graph, k, always_preserve, device):
    """
    Find the k-core of the graph
    Only isolate the low degree nodes by removing their edges instead of removing the nodes to maintain the same node - nodeID mapping

    Parameters:
        graph: Union[PyTorch Geometric HeteroData, PyTorch Geometric Data]

        k: int

        always_preserve: dictionary
            key=node_type, value=torch.tensor with the indices of the nodes to be preserved

    Returns:
        The k-core of the graph
    """

    try:
        if isinstance(graph, Data):
            # TODO: add support for homogenous graphs
            raise NotImplementedError

        # find the in-degree of every node
        in_degree = {
            ntype: torch.zeros(graph[ntype].x.size(0), dtype=torch.int).to(device)
            for ntype in graph.node_types
        }
        for edge_type in graph.edge_types:
            for tgt_node in graph[edge_type].edge_index[1]:
                in_degree[edge_type[2]][tgt_node] += 1

        # prune all the edges of high degree nodes except the edges of the nodes in always_preserve
        non_k_core_mask = {}
        for ntype in graph.node_types:
            non_k_core_mask[ntype] = (in_degree[ntype]) > 0 & (in_degree[ntype] < k)
            if always_preserve[ntype].shape[0] != 0:
                non_k_core_mask[ntype][always_preserve[ntype]] = False
        del in_degree

        for etype in graph.edge_types:
            src, dst = graph[etype].edge_index[0], graph[etype].edge_index[1]
            src_nodes_non_k_core = torch.arange(
                graph[etype[0]].x.size(0), device=device
            )[non_k_core_mask[etype[0]]]
            dst_nodes_non_k_core = torch.arange(
                graph[etype[2]].x.size(0), device=device
            )[non_k_core_mask[etype[2]]]
            non_k_core_edge_mask = torch.isin(src, src_nodes_non_k_core) | torch.isin(
                dst, dst_nodes_non_k_core
            )
            graph[etype].edge_index = graph[etype].edge_index[:, non_k_core_edge_mask]

        return graph

    except NotImplementedError as e:
        print(f"Support for homogenous graphs has not been added yet\n{e}")


def get_neg_path_score_func(graph, eweights, exclude_node):
    """
    Compute the negative path score for the shortest path algorithm

    Parameters:
        graph: a PyTorch HeteroData object

        eweights: Dict[str, Tensor]
            key=edge_type, value=torch.tensor with the weight of the edge at the corresponding index

        exclude_node: dictionary
            key=node_type, value=torch.tensor with the indices of the nodes to exclude

    Returns:
        neg_path_score_func: a callable function
            Takes in two node IDs and the edge type, and returns the edge weight
    """

    device = graph[graph.node_types[0]].x.device

    weights = {}
    if eweights is None:
        for etype in graph.edge_types:
            weights[etype] = torch.ones(graph[etype].edge_index.size(), device=device)
        eweights = weights
    del weights

    # calculate the log of in-degrees
    in_degree = {
        ntype: torch.zeros(graph[ntype].x.size(0), dtype=torch.int).to(device)
        for ntype in graph.node_types
    }
    for edge_type in graph.edge_types:
        for tgt_node in graph[edge_type].edge_index[1]:
            if not torch.any(torch.eq(exclude_node[edge_type[2]], tgt_node)):
                in_degree[edge_type[2]][tgt_node] += 1

    for ntype in in_degree:
        in_degree[ntype] = torch.log(torch.clamp(in_degree[ntype], min=1))

    log_eweights = {etype: torch.log(eweights[etype]) for etype in eweights}

    # calculate the path scores
    neg_path_score_map = {}
    for etype in graph.edge_types:
        edge_score_map = {}
        src, dst = graph[etype].edge_index[0], graph[etype].edge_index[1]
        for idx in range(graph[etype].edge_index.size(1)):
            edge_score_map[(src[idx].item(), dst[idx].item())] = (
                in_degree[etype[2]][dst[idx]] - log_eweights[etype][idx]
            )
        neg_path_score_map[etype] = edge_score_map

    def neg_path_score_func(u, v, etype):
        if (u, v) in neg_path_score_map[etype]:
            return neg_path_score_map[etype][(u, v)]
        else:
            return neg_path_score_map[etype][(v, u)]

    return neg_path_score_func


class PathBuffer:
    """
    For finding the shortest paths
    """

    def __init__(self):
        self.paths = set()
        self.sortedpaths = list()
        self.counter = count()

    def __len__(self):
        return len(self.sortedpaths)

    def push(self, cost, path):
        hashable_path = tuple(path)
        if hashable_path not in self.paths:
            heappush(self.sortedpaths, (cost, next(self.counter), path))
            self.paths.add(hashable_path)

    def pop(self):
        (cost, num, path) = heappop(self.sortedpaths)
        hashable_path = tuple(path)
        self.paths.remove(hashable_path)
        return path


def get_neighbors(
    graph, ntype, nid, ignore_nodes_init, ignore_edges_init, forward=True
):
    """
    Get all the neighbors of a given node

    Parameters:
        graph: a PyTorch HeteroData object

        ntype: string
            Source node type

        nid : int
            Source node id

        ignore_nodes_init : Set[Tuple[str, Tensor]]
            str=node_type, Tensor=nodeID

        ignore_edges_init :  Set[Tuple[str, Tensor]]
           str=edge_type, Tensor=edge_index

        forward: bool
            Consider the edge in its direction if True or in reverse if False

    Returns:
        neighbors: List[Tuple[int, str, str]]
            Neighbors of the node as (nid, ntype, etype) tuples
    """

    neighbors = []
    for etype in graph.edge_types:
        if "self" in etype[1]:
            continue
        if forward and etype[0] == ntype:
            indices = (graph[etype].edge_index[0] == nid).nonzero()
            edge_neighbors = graph[etype].edge_index[1][indices].squeeze()
            if edge_neighbors.shape == ():
                continue
            for n in edge_neighbors:
                if (etype[2], n) not in ignore_nodes_init and (
                    etype,
                    torch.tensor([[nid], [n.item()]]),
                ) not in ignore_edges_init:
                    neighbors.append((n.item(), etype[2], etype))
        elif not forward and etype[2] == ntype:
            indices = (graph[etype].edge_index[1] == nid).nonzero()
            edge_neighbors = graph[etype].edge_index[0][indices].squeeze()
            if edge_neighbors.shape == ():
                continue
            for n in edge_neighbors:
                if (etype[0], n) not in ignore_nodes_init and (
                    etype,
                    torch.tensor([[n.item()], [nid]]),
                ) not in ignore_edges_init:
                    neighbors.append((n.item(), etype[0], etype))
    return neighbors


def get_distance(dist, node):
    """
    Return dist[node][0] if present, else return infinity

    Parameters:
        dist: Dict[Tuple[int, str]]

        node: Tuple[int, str]
    """

    return dist[node][0] if node in dist else float("inf")


def bidirectional_dijkstra(
    graph,
    src_ntype,
    src_nid,
    tgt_ntype,
    tgt_nid,
    weight,
    ignore_nodes_init,
    ignore_edges_init,
):
    """
    Dijkstra's algorithm for shortest paths using bidirectional search

    Parameters:
        graph: a PyTorch HeteroData object

        src_ntype: string
            Source node type

        src_nid : int
            Source node id

        tgt_ntype: string
            Target node type

        tgt_nid : int
            Target node id

        weight: a callable function, optional
            Takes in two node Ids and the edge type and returns the weight

        k: int
            Number of paths

        ignore_nodes_init : Set[Tuple[str, Tensor]]
            str=node_type, Tensor=nodeID

        ignore_edges_init :  Set[Tuple[str, Tensor]]
           str=edge_type, Tensor=edge_index

    Returns:
        length : int
            Shortest path length
        path: List[Tuple[int, int, str]]
    """
    if src_nid == tgt_nid:
        return (0, [(src_nid, tgt_nid, "_")])

    Qf, Qb = PriorityQueue(), PriorityQueue()
    df, db = {}, {}
    Sf, Sb = set(), set()
    Qf.put((0, src_nid, src_ntype))
    Qb.put((0, tgt_nid, tgt_ntype))
    df[(src_nid, src_ntype)], db[(tgt_nid, tgt_ntype)] = (0, []), (0, [])
    mu, mu_path = float("inf"), []

    while not Qf.empty() and not Qb.empty():
        u, v = Qf.get(), Qb.get()
        Sf.add((u[1], u[2]))
        Sb.add((v[1], v[2]))
        u_neighbors = get_neighbors(
            graph, u[2], u[1], ignore_nodes_init, ignore_edges_init, forward=True
        )
        v_neighbors = get_neighbors(
            graph, v[2], v[1], ignore_nodes_init, ignore_edges_init, forward=False
        )

        u, v = (u[1], u[2]), (v[1], v[2])
        for t in u_neighbors:
            # relax
            x, e = (t[0], t[1]), t[2]
            if x not in Sf and get_distance(df, x) > get_distance(df, u) + weight(
                u[0], x[0], e
            ):
                df[x] = (df[u][0] + weight(u[0], x[0], e), df[u][1] + [(u[0], x[0], e)])
                Qf.put((df[x][0], t[0], t[1]))
            if (
                x in Sb
                and get_distance(df, u) + weight(u[0], x[0], e) + get_distance(db, x)
                < mu
            ):
                mu = df[u][0] + weight(u[0], x[0], e) + db[x][0]
                mu_path = df[u][1] + [(u[0], x[0], e)] + db[x][1][::-1]
        for t in v_neighbors:
            # relax
            x, e = (t[0], t[1]), t[2]
            if x not in Sb and get_distance(db, x) > get_distance(db, v) + weight(
                x[0], v[0], e
            ):
                db[x] = (db[v][0] + weight(x[0], v[0], e), db[v][1] + [(v[0], x[0], e)])
                Qb.put((db[x][0], t[0], t[1]))
            if (
                x in Sf
                and get_distance(db, v) + weight(x[0], v[0], e) + get_distance(df, x)
                < mu
            ):
                mu = db[v][0] + weight(x[0], v[0], e) + df[x][0]
                mu_path = df[x][1] + (db[v][1] + [(x[0], v[0], e)])[::-1]

        if get_distance(df, u) + get_distance(db, v) >= mu:
            break

    return mu, mu_path


def k_shortest_paths_generator(
    graph,
    src_ntype,
    src_nid,
    tgt_ntype,
    tgt_nid,
    weight=None,
    k=5,
    ignore_nodes_init=set(),
    ignore_edges_init=set(),
):
    """
    Generate atmost k simple paths in the graph from src_nid to tgt_nid
    each with maximum lenghth `max_length`, return starting from the shortest ones
    If a weighted shortest path search is to be used, no negative weights are allowed

    Parameters:
        graph: a PyTorch HeteroData object

        src_ntype: string
            Source node type

        src_nid : int
            Source node id

        tgt_ntype: string
            Target node type

        tgt_nid : int
            Target node id

        weight: a callable function, optional
            Takes in two node Ids and the edge type and returns the weight

        k: int
            Number of paths

        ignore_nodes_init : Set[Tuple[str, Tensor]]
            str=node_type, Tensor=nodeID

        ignore_edges_init :  Set[Tuple[str, Tensor]]
           str=edge_type, Tensor=edge_index

    Returns:
         path_generator: generator
            A generator that produces lists of tuples (path score, path), in order from
            shortest to longest, each path is a list of tuples (src_nid, dst_nid, etype)
    """

    device = graph[graph.node_types[0]].x.device

    if not weight:
        weight = lambda u, v, etype: 1

    def length_func(path):
        return sum(weight(u, v, etype) for (u, v, etype) in path)

    listA = list()
    listB = PathBuffer()
    prev_path = None

    while not prev_path or len(listA) < k:
        if not prev_path:
            length, path = bidirectional_dijkstra(
                graph,
                src_ntype,
                src_nid,
                tgt_ntype,
                tgt_nid,
                weight,
                ignore_nodes_init,
                ignore_edges_init,
            )
            listB.push(length, path)
        else:
            ignore_nodes = set(ignore_nodes_init) if ignore_nodes_init else set()
            ignore_edges = set(ignore_edges_init) if ignore_edges_init else set()
            for i in range(1, len(prev_path)):
                root = prev_path[:i]
                root_length = length_func(root)
                for path in listA:
                    if path[:i] == root:
                        ignore_edges.add(
                            (
                                path[i - 1][2],
                                torch.tensor(
                                    [[path[i - 1][0]], [path[i - 1][1]]], device=device
                                ),
                            )
                        )

                try:
                    length, spur = bidirectional_dijkstra(
                        graph,
                        src_ntype,
                        root[-1][1],
                        tgt_ntype,
                        tgt_nid,
                        weight,
                        ignore_nodes_init=ignore_nodes,
                        ignore_edges_init=ignore_edges,
                    )
                    path = root[:-1] + spur
                    listB.push(root_length + length, path)
                except ValueError:
                    pass
                ignore_nodes.add((root[-1][2][2], root[-1][1]))

        if listB:
            path = listB.pop()
            yield path
            listA.append(path)
            if len(path) == 0:
                path = None
            prev_path = path
        else:
            break


def k_shortest_paths_with_max_length(
    graph,
    src_ntype,
    src_nid,
    tgt_ntype,
    tgt_nid,
    weight=None,
    k=5,
    max_length=None,
    ignore_nodes=set(),
    ignore_edges=set(),
):
    """
    Generate at most k simple paths in the graph from src_nid to tgt_nid,
    each with maximum lenghth max_length, return starting from the shortest ones
    If a weighted shortest path search is to be used, no negative weights are allowed

    Parameters:
       See function k_shortest_paths_generator

    Returns:
        paths: list of lists
            Each list is a path containing (src_nid, dst_nid, etype)
    """

    path_generator = k_shortest_paths_generator(
        graph,
        src_ntype,
        src_nid,
        tgt_ntype,
        tgt_nid,
        weight=weight,
        k=k,
        ignore_nodes_init=ignore_nodes,
        ignore_edges_init=ignore_edges,
    )

    try:
        if max_length:
            paths = [path for path in path_generator if len(path) <= max_length + 1]
        else:
            paths = list(path_generator)

    except ValueError:
        paths = [[]]

    return paths
