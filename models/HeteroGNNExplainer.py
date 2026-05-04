import torch
import torch.nn as nn
from collections import defaultdict
import tqdm
from utils.Explainer_utils import get_edge_mask_dict, hetero_src_tgt_khop_in_subgraph


class HeteroGNNExplainer(nn.Module):
    """
    GNNExplainer for heterogenous link prediction explanation

    Migrated to PyTorch Geometric from the DGL implementation in https://github.com/amazon-science/page-link-path-based-gnn-explanation/blob/main/baselines/baseline_explainer.py

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
        log: bool, optional
            If True, the computation process will be logged, defaults to False
    """

    def __init__(
        self,
        model,
        src_ntype,
        tgt_ntype,
        lr=0.01,
        num_epochs=100,
        alpha1=0.005,
        alpha2=1.0,
        log=False,
    ):
        super(HeteroGNNExplainer, self).__init__()
        self.model = model
        self.src_ntype = src_ntype
        self.tgt_ntype = tgt_ntype

        self.lr = lr
        self.num_epochs = num_epochs
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.log = log

        self.all_loss = defaultdict(list)

    def _init_masks(self, graph, device):
        """
        Initialize learnable edge masks

        Parameters:
            graph: a PyTorch HeteroData object

        Returns:
            edge_mask_dict: dictionary
                key=etype, value=torch.nn.Parameter with size number of etype edges
        """

        return get_edge_mask_dict(graph, device)

    def _loss_regularize(self, loss, eweights):
        """
        Add regularization terms to the loss

        Parameters:
            loss: Tensor
                The loss value
            eweights: Tensor
                Edge mask of shape E, where E is the number of edges

        Returns:
            Tensor
                The loss value which includes the regularization terms
        """

        # epsilon for numerical stability
        eps = 1e-15

        # edge mask sparsity regularization
        reg1 = torch.sum(eweights)
        loss += self.alpha1 * reg1

        # edge mask entropy regularization
        ent = -eweights * torch.log(eweights * eps) - (1 - eweights) * torch.log(
            1 - eweights + eps
        )
        reg2 = ent.mean()
        loss += self.alpha2 * reg2

        self.all_loss["reg1"] += [reg1.item()]
        self.all_loss["reg2"] += [reg2.item()]

        return loss

    def get_comp_g_edge_mask(self, edge_type, src_nid, tgt_nid, graph, device):
        """
        Get the explanation mask for the computation graph

        Parameters:
            src_nid: int
                The source node of the link
            tgt_nid: int
                The target node of the link
            etype: Tuple[str, str, str]
                The edge type of the link to predict
            graph: PyTorch Geometric HeteroData object
                The heterogenous graph

        Returns:
            edge_mask_dict: dictionary
                key=etype, value=torch.nn.Parameter with size number of etype edges
        """

        graph[edge_type].edge_label_index = torch.tensor([[src_nid], [tgt_nid]])

        edge_mask_dict = self._init_masks(graph, device)
        optimizer = torch.optim.Adam(edge_mask_dict.values(), lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.num_epochs)

        eweight_norm = 0
        for epoch in range(1, self.num_epochs + 1):
            # apply sigmoid to edge_mask to get eweight
            eweight_dict = {
                etype: edge_mask_dict[etype].sigmoid() for etype in edge_mask_dict
            }

            # get the initial prediction
            self.model.eval()
            with torch.no_grad():
                score = self.model(graph, eweights=eweight_dict)
                pred = score > 0  # .int().item()

            loss = -(1**pred) * score.sigmoid().log()

            self.all_loss["loss"] += [loss.item()]

            eweights = torch.cat(list(edge_mask_dict.values())).sigmoid()

            curr_eweight_norm = eweights.norm()
            eweight_norm = curr_eweight_norm

            loss = self._loss_regularize(loss, eweights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.all_loss["total_loss"] += [loss.item()]

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        return edge_mask_dict

    def explain(self, src_nid, tgt_nid, graph, edge_type, device, num_hops=3):
        """
        Compute the explanation subgraph

        Parameters:
            src_nid: int
                The source node of the link
            tgt_nid: int
                The target node of the link
            graph: PyTorch Geometric HeteroData object
                The heterogenous graph
            edge_type: Tuple[str, str, str]
                The predicted link's type

        Returns:
            edge_mask_dict: dictionary
                key=etype, value=torch.nn.Parameter with size number of etype edges
        """

        # Extract the computation graph (k-hop subgraph)

        (comp_g_src_nid, comp_g_tgt_nid, comp_g, mapping) = (
            hetero_src_tgt_khop_in_subgraph(
                self.src_ntype,
                src_nid.item(),
                self.tgt_ntype,
                tgt_nid.item(),
                graph,
                num_hops,
                device,
            )
        )

        edge_mask_dict = self.get_comp_g_edge_mask(
            edge_type, comp_g_src_nid, comp_g_tgt_nid, comp_g, device
        )

        return edge_mask_dict
