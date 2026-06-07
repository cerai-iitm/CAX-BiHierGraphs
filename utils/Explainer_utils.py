import torch
from torch_geometric.data import Data
import numpy as np
from itertools import count
from heapq import heappop, heappush
from queue import PriorityQueue


def get_edge_mask_dict(graph, device):
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
    # BUG FIX: all masks must be on `device` to match graph edge indices
    node_mask = {
        ntype: torch.zeros(graph[ntype].x.size(0), dtype=torch.bool, device=device)
        for ntype in graph.node_types
    }
    edge_mask = {
        etype: torch.empty(graph[etype].edge_index.size(1), dtype=torch.bool, device=device)
        for etype in graph.edge_types
        if "self" not in etype[1]
    }

    subset = []
    subset_zero_hop = {}
    for ntype in graph.node_types:
        if ntype in center_nodes:
            # BUG FIX: keep on device (was forced to CPU with .cpu())
            subset_zero_hop[ntype] = center_nodes[ntype].to(device)
        else:
            # BUG FIX: empty tensors must also be on device
            subset_zero_hop[ntype] = torch.tensor([], dtype=torch.long, device=device)
    subset.append(subset_zero_hop)
    del subset_zero_hop

    for _ in range(k):
        for ntype in node_mask:
            node_mask[ntype].fill_(False)
        for ntype in node_mask:
            if subset[-1][ntype].numel():
                node_mask[ntype][subset[-1][ntype]] = True

        subset_hop = {}
        for etype in graph.edge_types:
            if "self" in etype[1]:
                continue

            ei = graph[etype].edge_index

            if not edge_weights:
                torch.index_select(node_mask[etype[0]], 0, ei[0], out=edge_mask[etype])
                neighbors = ei[1][edge_mask[etype]]
                dst_type = etype[2]
                subset_hop[dst_type] = (
                    torch.cat((subset_hop[dst_type], neighbors), dim=0)
                    if dst_type in subset_hop
                    else neighbors
                )

            else:
                ew = edge_weights[etype]

                active_dst_mask = node_mask[etype[2]][ei[1]]
                active_ei_src = ei[0][active_dst_mask]
                active_ei_dst = ei[1][active_dst_mask]
                active_weights = ew[
                    active_dst_mask[: ew.size(0)]
                    if ew.size(0) < active_dst_mask.size(0)
                    else active_dst_mask
                ]

                if active_ei_src.numel() == 0:
                    continue
                unique_dsts = active_ei_dst.unique()
                selected_sources = []
                for dst in unique_dsts:
                    dst_mask = active_ei_dst == dst
                    srcs = active_ei_src[dst_mask]
                    wts = active_weights[dst_mask]
                    if srcs.numel() <= num_neighbors:
                        selected_sources.append(srcs)
                    else:
                        topk_idx = torch.topk(wts, num_neighbors, dim=0)[1]
                        selected_sources.append(srcs[topk_idx])

                neighbors = torch.cat(selected_sources, dim=0)
                src_type = etype[0]
                subset_hop[src_type] = (
                    torch.cat((subset_hop[src_type], neighbors), dim=0)
                    if src_type in subset_hop
                    else neighbors
                )

        for ntype in graph.node_types:
            if ntype not in subset_hop:
                # BUG FIX: empty fallback tensors must also be on device
                subset_hop[ntype] = torch.tensor([], dtype=torch.long, device=device)

        subset.append(subset_hop)
        del subset_hop

    subgraph_dict = {}
    for ntype in graph.node_types:
        nodes = [sh[ntype] for sh in subset]
        subgraph_dict[ntype] = torch.cat(nodes, dim=0).unique()
    del subset

    mapping = {}
    for ntype in subgraph_dict:
        mapping[ntype] = {
            node.item(): idx for idx, node in enumerate(subgraph_dict[ntype])
        }

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
    src_nid = src_nid.item() if torch.is_tensor(src_nid) else src_nid
    tgt_nid = tgt_nid.item() if torch.is_tensor(tgt_nid) else tgt_nid

    if src_ntype == tgt_ntype:
        pred_dict = {
            src_ntype: torch.tensor([src_nid, tgt_nid], dtype=torch.long, device=device)
        }
    else:
        pred_dict = {
            src_ntype: torch.tensor([src_nid], dtype=torch.long, device=device),
            tgt_ntype: torch.tensor([tgt_nid], dtype=torch.long, device=device),
        }

    sghetero, mapping = hetero_k_hop_subgraph(
        graph, pred_dict, k, device, edge_weights, num_neighbors
    )
    sghetero_src_nid = mapping[src_ntype][src_nid]
    sghetero_tgt_nid = mapping[tgt_ntype][tgt_nid]

    return sghetero_src_nid, sghetero_tgt_nid, sghetero, mapping


def remove_edges_of_high_degree_nodes(graph, max_degree, always_preserve, device):
    try:
        if isinstance(graph, Data):
            raise NotImplementedError

        # BUG FIX: in_degree tensors must be on device to match edge indices
        in_degree = {
            ntype: torch.zeros(graph[ntype].x.size(0), dtype=torch.int, device=device)
            for ntype in graph.node_types
        }
        for edge_type in graph.edge_types:
            dst = graph[edge_type].edge_index[1]
            in_degree[edge_type[2]].scatter_add_(
                0, dst, torch.ones_like(dst, dtype=torch.int)
            )

        high_degree_mask = {}
        for ntype in graph.node_types:
            high_degree_mask[ntype] = in_degree[ntype] > max_degree
            high_degree_mask[ntype][always_preserve[ntype]] = False
        del in_degree

        for etype in graph.edge_types:
            src, dst = graph[etype].edge_index
            n_src = graph[etype[0]].x.size(0)
            n_dst = graph[etype[2]].x.size(0)
            # BUG FIX: arange must be on device to match high_degree_mask
            src_hd = torch.arange(n_src, device=device)[high_degree_mask[etype[0]]]
            dst_hd = torch.arange(n_dst, device=device)[high_degree_mask[etype[2]]]
            keep = ~(torch.isin(src, src_hd) | torch.isin(dst, dst_hd))
            graph[etype].edge_index = graph[etype].edge_index[:, keep]

        return graph

    except NotImplementedError as e:
        print(f"Support for homogenous graphs has not been added yet\n{e}")


def remove_edges_except_k_core_graph(graph, k, always_preserve, device):
    try:
        if isinstance(graph, Data):
            raise NotImplementedError

        # BUG FIX: in_degree tensors must be on device to match edge indices
        in_degree = {
            ntype: torch.zeros(graph[ntype].x.size(0), dtype=torch.int, device=device)
            for ntype in graph.node_types
        }
        for edge_type in graph.edge_types:
            dst = graph[edge_type].edge_index[1]
            in_degree[edge_type[2]].scatter_add_(
                0, dst, torch.ones_like(dst, dtype=torch.int)
            )

        non_k_core_mask = {}
        for ntype in graph.node_types:
            deg = in_degree[ntype]
            non_k_core_mask[ntype] = (deg > 0) & (deg < k)
            if always_preserve[ntype].shape[0] != 0:
                non_k_core_mask[ntype][always_preserve[ntype]] = False
        del in_degree

        for etype in graph.edge_types:
            src, dst = graph[etype].edge_index
            n_src = graph[etype[0]].x.size(0)
            n_dst = graph[etype[2]].x.size(0)
            # BUG FIX: arange must be on device to match non_k_core_mask
            src_nkc = torch.arange(n_src, device=device)[non_k_core_mask[etype[0]]]
            dst_nkc = torch.arange(n_dst, device=device)[non_k_core_mask[etype[2]]]
            keep = ~(torch.isin(src, src_nkc) | torch.isin(dst, dst_nkc))
            graph[etype].edge_index = graph[etype].edge_index[:, keep]

        return graph

    except NotImplementedError as e:
        print(f"Support for homogenous graphs has not been added yet\n{e}")


# ── Everything below is CPU-only path/graph algorithms — no device needed ────

def get_neg_path_score_func(graph, eweights, exclude_node):
    device = graph[graph.edge_types[0]].edge_index.device

    if eweights is None:
        eweights = {
            etype: torch.ones(graph[etype].edge_index.size(1), device=device)
            for etype in graph.edge_types
        }
    else:
        eweights = {etype: w.to(device) for etype, w in eweights.items()}

    in_degree = {
        ntype: torch.zeros(graph[ntype].x.size(0), dtype=torch.float, device=device)
        for ntype in graph.node_types
    }
    for edge_type in graph.edge_types:
        dst = graph[edge_type].edge_index[1]
        excl = exclude_node.get(edge_type[2], torch.tensor([], dtype=torch.long, device=device)).to(device)
        excluded_mask = torch.isin(dst, excl)
        contrib = (~excluded_mask).float()
        in_degree[edge_type[2]].scatter_add_(0, dst, contrib)

    for ntype in in_degree:
        in_degree[ntype] = torch.log(in_degree[ntype].clamp(min=1))

    log_eweights = {etype: torch.log(eweights[etype]) for etype in eweights}

    neg_path_score_map = {}
    for etype in graph.edge_types:
        edge_score_map = {}
        src_nodes = graph[etype].edge_index[0]
        dst_nodes = graph[etype].edge_index[1]
        for idx in range(graph[etype].edge_index.size(1)):
            s, d = src_nodes[idx].item(), dst_nodes[idx].item()
            edge_score_map[(s, d)] = (
                in_degree[etype[2]][d] - log_eweights[etype][idx]
            )
        neg_path_score_map[etype] = edge_score_map

    def neg_path_score_func(u, v, etype):
        if (u, v) in neg_path_score_map[etype]:
            return neg_path_score_map[etype][(u, v)]
        return neg_path_score_map[etype][(v, u)]

    return neg_path_score_func


class PathBuffer:
    def __init__(self):
        self.paths = set()
        self.sortedpaths = list()
        self.counter = count()

    def __len__(self):
        return len(self.sortedpaths)

    def push(self, cost, path):
        hashable_path = tuple(tuple(e) for e in path)
        if hashable_path not in self.paths:
            heappush(self.sortedpaths, (cost, next(self.counter), path))
            self.paths.add(hashable_path)

    def pop(self):
        cost, num, path = heappop(self.sortedpaths)
        hashable_path = tuple(tuple(e) for e in path)
        self.paths.remove(hashable_path)
        return path


def get_neighbors(graph, ntype, nid, ignore_nodes_init, ignore_edges_init, forward=True):
    neighbors = []
    for etype in graph.edge_types:
        if "self" in etype[1]:
            continue
        if forward and etype[0] == ntype:
            indices = (graph[etype].edge_index[0] == nid).nonzero(as_tuple=False).squeeze(1)
            if indices.numel() == 0:
                continue
            edge_neighbors = graph[etype].edge_index[1][indices]
            for n in edge_neighbors:
                nval = n.item()
                if (etype[2], nval) not in ignore_nodes_init and (
                    etype, nid, nval
                ) not in ignore_edges_init:
                    neighbors.append((nval, etype[2], etype))
        elif not forward and etype[2] == ntype:
            indices = (graph[etype].edge_index[1] == nid).nonzero(as_tuple=False).squeeze(1)
            if indices.numel() == 0:
                continue
            edge_neighbors = graph[etype].edge_index[0][indices]
            for n in edge_neighbors:
                nval = n.item()
                if (etype[0], nval) not in ignore_nodes_init and (
                    etype, nval, nid
                ) not in ignore_edges_init:
                    neighbors.append((nval, etype[0], etype))
    return neighbors


def get_distance(dist, node):
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
    if src_nid == tgt_nid:
        return (0, [(src_nid, tgt_nid, "_")])

    Qf, Qb = PriorityQueue(), PriorityQueue()
    df, db = {}, {}
    Sf, Sb = set(), set()
    Qf.put((0, src_nid, src_ntype))
    Qb.put((0, tgt_nid, tgt_ntype))
    df[(src_nid, src_ntype)] = (0, None, None, None)
    db[(tgt_nid, tgt_ntype)] = (0, None, None, None)
    mu, mu_meet = float("inf"), None

    def reconstruct(meet_fwd, meet_bwd, via_edge):
        fwd_path = []
        node = meet_fwd
        while df[node][1] is not None:
            _, par_nid, par_ntype, etype = df[node]
            fwd_path.append((par_nid, node[0], etype))
            node = (par_nid, par_ntype)
        fwd_path.reverse()
        bwd_path = []
        node = meet_bwd
        while db[node][1] is not None:
            _, par_nid, par_ntype, etype = db[node]
            bwd_path.append((node[0], par_nid, etype))
            node = (par_nid, par_ntype)
        return fwd_path + [via_edge] + bwd_path

    while not Qf.empty() and not Qb.empty():
        uf_dist, uf_nid, uf_ntype = Qf.get()
        ub_dist, ub_nid, ub_ntype = Qb.get()
        u = (uf_nid, uf_ntype)
        v = (ub_nid, ub_ntype)
        Sf.add(u)
        Sb.add(v)

        u_neighbors = get_neighbors(graph, uf_ntype, uf_nid, ignore_nodes_init, ignore_edges_init, forward=True)
        v_neighbors = get_neighbors(graph, ub_ntype, ub_nid, ignore_nodes_init, ignore_edges_init, forward=False)

        for (t_nid, t_ntype, e) in u_neighbors:
            x = (t_nid, t_ntype)
            new_dist = df[u][0] + weight(uf_nid, t_nid, e)
            if x not in Sf and new_dist < get_distance(df, x):
                df[x] = (new_dist, uf_nid, uf_ntype, e)
                Qf.put((new_dist, t_nid, t_ntype))
            if x in Sb:
                total = new_dist + db[x][0]
                if total < mu:
                    mu = total
                    mu_meet = (u, x, (uf_nid, t_nid, e))

        for (t_nid, t_ntype, e) in v_neighbors:
            x = (t_nid, t_ntype)
            new_dist = db[v][0] + weight(t_nid, ub_nid, e)
            if x not in Sb and new_dist < get_distance(db, x):
                db[x] = (new_dist, ub_nid, ub_ntype, e)
                Qb.put((new_dist, t_nid, t_ntype))
            if x in Sf:
                total = db[v][0] + weight(t_nid, ub_nid, e) + df[x][0]
                if total < mu:
                    mu = total
                    mu_meet = (x, v, (t_nid, ub_nid, e))

        if df[u][0] + db[v][0] >= mu:
            break

    if mu_meet is None:
        raise ValueError("No path found")

    fwd_node, bwd_node, bridge = mu_meet
    path = reconstruct(fwd_node, bwd_node, bridge)
    return mu, path


def k_shortest_paths_generator(
    graph,
    src_ntype,
    src_nid,
    tgt_ntype,
    tgt_nid,
    weight=None,
    k=5,
    ignore_nodes_init=None,
    ignore_edges_init=None,
):
    if ignore_nodes_init is None:
        ignore_nodes_init = set()
    if ignore_edges_init is None:
        ignore_edges_init = set()

    if weight is None:
        weight = lambda u, v, etype: 1

    def length_func(path):
        return sum(weight(u, v, etype) for (u, v, etype) in path)

    listA = []
    listB = PathBuffer()
    prev_path = None

    while not prev_path or len(listA) < k:
        if not prev_path:
            length, path = bidirectional_dijkstra(
                graph, src_ntype, src_nid, tgt_ntype, tgt_nid,
                weight, ignore_nodes_init, ignore_edges_init,
            )
            listB.push(length, path)
        else:
            ignore_nodes = set(ignore_nodes_init)
            ignore_edges = set(ignore_edges_init)
            for i in range(1, len(prev_path)):
                root = prev_path[:i]
                root_length = length_func(root)
                for path in listA:
                    if path[:i] == root:
                        pe = path[i - 1]
                        ignore_edges.add((pe[2], pe[0], pe[1]))

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
            prev_path = path if path else None
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
    ignore_nodes=None,
    ignore_edges=None,
):
    if ignore_nodes is None:
        ignore_nodes = set()
    if ignore_edges is None:
        ignore_edges = set()

    path_generator = k_shortest_paths_generator(
        graph, src_ntype, src_nid, tgt_ntype, tgt_nid,
        weight=weight, k=k,
        ignore_nodes_init=ignore_nodes,
        ignore_edges_init=ignore_edges,
    )

    try:
        if max_length:
            paths = [p for p in path_generator if len(p) <= max_length + 1]
        else:
            paths = list(path_generator)
    except ValueError:
        paths = [[]]

    return paths