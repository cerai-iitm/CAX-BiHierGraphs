import torch
import torch.nn as nn
from collections import defaultdict
import tqdm
from Explainer_utils import get_edge_mask_dict, remove_edges_of_high_degree_nodes, remove_edges_except_k_core_graph, get_neg_path_score_func, k_shortest_paths_with_max_length, hetero_src_tgt_khop_in_subgraph

class PaGELink(nn.Module):
    '''
    Path-based GNN Explanation for Heterogeneous Link Prediction (PaGE-Link)
    
    Migrated to PyTorch Geometric from the DGL implementation in https://github.com/amazon-science/page-link-path-based-gnn-explanation/blob/main/explainer.py
    
    Parameters:
        model: nn.Module
            The GNN model to explain
        src_ntype: str
            The predicted link's source node's type
        tgt_ntype: str
            The predicted link's destination node's type
        lr: float, optional
            The learning rate to use, defaults to 0.01
        num_epochs: int, optional
            The number of epochs to train the explainer for, defaults to 100
        alpha1: float, optional
            A higher value will make the explanation edge masks more sparse by decreasing
            the sum of the edge mask, defaults to 0.005
        alpha2: float, optional
            A higher value will make the explanation edge masks more discrete by decreasing
            the entropy of the edge mask, defaults to 1.0
        alpha : float, optional
            A higher value will make edges on high-quality paths to have higher weights, defaults to 1.0
        beta : float, optional
            A higher value will make edges off high-quality paths to have lower weights, defaults to 1.0
        log: bool, optional
            If True, the computation process will be logged, defaults to False
    '''
    
    def __init__(self, model, src_ntype, tgt_ntype, lr=0.01, num_epochs=100, alpha1=0.005, alpha2=1.0, alpha=1.0, beta=1.0, log=False):
        super(PaGELink, self).__init__()
        self.model = model
        self.src_ntype = src_ntype
        self.tgt_ntype = tgt_ntype
        
        self.lr = lr
        self.num_epochs = num_epochs
        self.alpha1 = alpha1
        self.aplha2 = alpha2
        self.alpha = alpha
        self.beta = beta
        self.log = log
        
        self.all_loss = defaultdict(list)
        
    def _init_masks(self, graph, device):
        '''
        Initialize learnable edge masks
        
        Parameters:
            graph: a PyTorch HeteroData object
        
        Returns:
            edge_mask_dict: dictionary
                key=etype, value=torch.nn.Parameter with size number of etype edges
        '''
        
        return get_edge_mask_dict(graph, device)
    
    def prune_graph(self, graph, always_preserve, prune_max_degree=-1, k_core=2):
        '''
        Prune edges by (optionally) removing edges of high degree nodes and extracting k-core
        
        Parameters:
            graph: a PyTorch HeteroData object
                The heterogenous graph
            always_preserve: dictionary
                key=node_type, value=torch.tensor with the indices of the nodes to be preserved
            prune_max_degree: int
                highest degree allowed for all nodes
                
            k_core: int
            
        Returns:
            pruned_graph: a PyTorch HeteroData object
        '''
        
        device = graph[graph.node_types[0]].x.device
        
        pruned_graph = None
        if prune_max_degree > 0:
            # remove edges of the high degree nodes
            max_degree_pruned_graph = remove_edges_of_high_degree_nodes(graph, prune_max_degree, always_preserve, device)
            del graph
            
            # k-core pruning
            k_core_pruned_graph = remove_edges_except_k_core_graph(max_degree_pruned_graph, k_core, always_preserve, device)
            
            if k_core_pruned_graph.num_edges <= 0: # no k-core found
                pruned_graph = max_degree_pruned_graph
                del k_core_pruned_graph, max_degree_pruned_graph
            else:
                pruned_graph = k_core_pruned_graph
                del k_core_pruned_graph, max_degree_pruned_graph
        else:
            # k-core pruning
            k_core_pruned_graph = remove_edges_except_k_core_graph(graph, k_core, always_preserve, device)
            
            if k_core_pruned_graph.num_edges <= 0: # no k-core found
                pruned_graph = graph
                del k_core_pruned_graph, graph
            else:
                pruned_graph = k_core_pruned_graph
                del k_core_pruned_graph, graph
                
        return pruned_graph
    
    def path_loss(self, src_nid, tgt_nid, graph, eweights=None, num_paths=5):
        '''
        Compute the path loss
        
        Parameters:
            src_nid: int
                The source node's ID
            tgt_nid: int
                The target node's ID
            graph: a PyTorch HeteroData object
                The heterogenous graph
            eweights: Tensor
                Edge weights considered while doing message passing
            num_paths: int
                Number of paths to compute the loss on
        
        Returns:
            loss: Tensor
                The path loss 
        '''
        
        device = graph[graph.node_types[0]].x.device
        
        exclude_nodes = {ntype: torch.tensor([], device=device) for ntype in graph.node_types}
        neg_path_score_func = get_neg_path_score_func(graph, eweights, exclude_nodes)
        paths = k_shortest_paths_with_max_length(graph,
                                                 self.src_ntype,
                                                 src_nid,
                                                 self.tgt_ntype,
                                                 tgt_nid,
                                                 weight=neg_path_score_func,
                                                 k=num_paths)
        
        # compute the loss
        all_path_edges = set([edge for path in paths for edge in path])
        loss_on_path, loss_off_path = 0.0, 0.0
        for etype in graph.edge_types:
            src, dst = graph[etype].edge_index[0], graph[etype].edge_index[1]
            for idx in range(src.shape[0]):
                if (src[idx].item(), dst[idx].item(), etype) in all_path_edges:
                    loss_on_path += neg_path_score_func(src[idx].item(), dst[idx].item(), etype)
                else:
                    loss_off_path += neg_path_score_func(src[idx].item(), dst[idx].item(), etype) 
                    
        self.all_loss['loss_on_path'] += [float(loss_on_path)]
        self.all_loss['loss_off_path'] += [float(loss_off_path)]
        
        loss = self.alpha * loss_on_path + self.beta * loss_off_path
        return torch.tensor([loss])  
        
    def get_edge_mask(self, graph, src_nid, tgt_nid, edge_type, prune_max_degree=-1, k_core=1, prune_graph=True, with_path_loss=True):
        '''
        Learn the edge mask dictionary
        
        Parameters:
            See the explain function
            
        Returns:
            edge_mask_dict: dictionary
                key=etype, value=torch.nn.Parameter with size number of etype edges
        '''
        
        self.model.eval()
        device = graph[graph.node_types[0]].x.device
        
        graph[edge_type].edge_label_index = torch.tensor([[src_nid], [tgt_nid]], device=device)
        
        always_preserve = {ntype: torch.tensor([], device=device) for ntype in graph.node_types}
        always_preserve[self.src_ntype] = torch.tensor([src_nid], device=device)
        always_preserve[self.tgt_ntype] = torch.tensor([tgt_nid], device=device)
        if prune_graph:
            graph = self.prune_graph(graph,
                                     always_preserve=always_preserve,
                                     prune_max_degree=prune_max_degree,
                                     k_core=k_core)
            
        edge_mask_dict = self._init_masks(graph, device=device)
        optimizer = torch.optim.Adam(edge_mask_dict.values(), lr=self.lr)
        
        if self.log:
            pbar = tqdm(total=self.num_epochs)
            
        eweight_norm = 0
        EPS = 1e-3
        for epoch in range(self.num_epochs):
            eweight_dict = {etype: edge_mask_dict[etype].sigmoid() for etype in edge_mask_dict}
            
            with torch.no_grad():
                score = self.model(graph, eweights=eweight_dict)
                pred = (score > 0)
                
            pred_loss = (-1) ** pred * score.sigmoid().log()
            self.all_loss['pred_loss'] += [pred_loss.item()]
            
            # Check for early stop
            curr_eweight_norm = torch.cat(list(edge_mask_dict.values())).norm()
            if abs(eweight_norm - curr_eweight_norm) < EPS:
                break
            eweight_norm = curr_eweight_norm
            
            path_loss = 0
            if with_path_loss:
                path_loss = self.path_loss(src_nid, tgt_nid, graph, eweights=eweight_dict)
                
            loss = pred_loss.to(device) + path_loss.to(device)
            loss.requires_grad = True
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self.all_loss['total_loss'] += [loss.item()]

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()
            
        edge_mask_dict = {k : v.detach() for k, v in edge_mask_dict.items()}
        return edge_mask_dict 
    
    def get_paths(self, src_nid, tgt_nid, graph, edge_mask_dict, num_paths=2, max_path_length=7):
        '''
        A post-processing step to turn the edge mask into paths
        
        Parameters:
            edge_mask_dict: dictionary
                key=etype, value=torch.nn.Parameter with size number of etype edges
                
            For others, see the explain function
            
        Returns:
            paths: list of lists
                each list contains Tuple[int, Tuple[str, str, str], int] units which are node1, etype, node2 for one hop
        '''
        
        device = graph[graph.node_types[0]].x.device
        
        eweight_dict = {etype: edge_mask_dict[etype].sigmoid() for etype in edge_mask_dict}
        
        exclude_nodes = {ntype: torch.tensor([], device=device) for ntype in graph.node_types}
        neg_path_score_func = get_neg_path_score_func(graph, eweight_dict, exclude_nodes)
        paths = k_shortest_paths_with_max_length(graph, self.src_ntype, src_nid,
                                                 self.tgt_ntype, tgt_nid, neg_path_score_func,
                                                 num_paths, max_path_length)
        
        if len(paths) == 0:
            # rare case where no paths are found, take the top edges
            cat_edge_mask = torch.cat([v for v in edge_mask_dict.values()])
            M = len(cat_edge_mask)
            k = min(num_paths * max_path_length, M)
            threshold = cat_edge_mask.topk(k)[0][-1].item()
            path = []
            for etype in edge_mask_dict:
                u, v = graph[etype].edge_index[0], graph[etype].edge_index[1]
                topk_edge_mask = edge_mask_dict[etype] >= threshold
                path += list(zip(u[topk_edge_mask].tolist(), 
                                 [etype] * topk_edge_mask.sum().item(),
                                 v[topk_edge_mask].tolist()))
            paths = [path]
        return paths
                
    def explain(self, src_nid, tgt_nid, graph, edge_type, num_hops=3, prune_max_degree=-1, k_core=3, num_paths=5, max_path_length=7, prune_graph=True, with_path_loss=True):
        '''
        Return the path explanations
        
        Parameters:
            src_nid: int
                The source node's ID
            tgt_nid: int
                The target node's ID
            graph: a PyTorch HeteroData object
                The heterogenous graph
            edge_type: Tuple[str, str, str]
                The predicted link's type
            num_hops: int
                Number of hops
            prune_max_degree: int
                Max degree allowed, if negative, no pruning is done
            k_core: int
            num_paths: int
                Number of paths to be generated
            max_path_length: int
            prune_graph: bool
                If True, the graph is pruned
            with_path_loss: bool
                If True, the path loss is also considered during backprop
                
        Returns:
            paths: list of lists
                each list contains Tuple[int, Tuple[str, str, str], int] units which are node1, etype, node2 for one hop
        '''
        
        device = graph[graph.node_types[0]].x.device
        
        # Extract the computation graph (k-hop subgraph)
        
        (comp_g_src_nid,
         comp_g_tgt_nid,
         comp_g,
         mapping) = hetero_src_tgt_khop_in_subgraph(self.src_ntype,
                                                    src_nid.item(),
                                                    self.tgt_ntype,
                                                    tgt_nid.item(),
                                                    graph,
                                                    num_hops,
                                                    device)
         
        edge_mask_dict = self.get_edge_mask(comp_g,
                                            comp_g_src_nid,
                                            comp_g_tgt_nid,
                                            edge_type,
                                            prune_max_degree,
                                            k_core,
                                            prune_graph,
                                            with_path_loss)
        
        comp_g_paths = self.get_paths(comp_g_src_nid,
                                      comp_g_tgt_nid,
                                      comp_g,
                                      edge_mask_dict,
                                      num_paths,
                                      max_path_length)
         
        return comp_g_paths