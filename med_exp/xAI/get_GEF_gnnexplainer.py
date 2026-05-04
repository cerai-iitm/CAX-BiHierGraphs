import pickle, os, math
import torch
import torch_geometric.transforms as T
from torch_geometric import seed_everything
import torch.nn.functional as F
from med_exp.graph.gnn_med_graph import Model
from HeteroGNNExplainer import HeteroGNNExplainer
from Explainer_utils import hetero_src_tgt_khop_in_subgraph

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
seed_everything(4321)

if __name__ == "__main__":
    noDupEdges = True

    if noDupEdges:
        hier_graph = pickle.load(open("med_exp/graph/MedGraph_NoDupEdges.pkl", "rb"))
    else:
        hier_graph = pickle.load(open("med_exp/graph/MedGraph.pkl", "rb"))

    hidden_dim = 64
    layers = 3

    # retrieve the model
    model_name = f"gnn_hier_med_graph_{layers}GATConv"
    model = Model(hier_graph, hier_graph["notes"].x.shape[1], hidden_dim, layers)
    model.to(device)

    if noDupEdges:
        model.load_state_dict(
            torch.load(
                "med_exp/graph/results/NoDupEdges/hetero_gnn_model_neg_10_NoDupEdges.pt"
            )
        )
    else:
        model.load_state_dict(
            torch.load("med_exp/graph/results/DupEdges/hetero_gnn_model_neg_10.pt")
        )
    model.eval()

    # initialize the explainer
    gnn_explainer = HeteroGNNExplainer(
        model=model, src_ntype="notes", tgt_ntype="icds", num_epochs=50
    ).to(device)

    if noDupEdges:
        test_data = pickle.load(
            open(
                "med_exp/graph/results/explanations/hetero_gnn_explainer/NoDupEdges/test_graph.pkl",
                "rb",
            )
        ).to(device)
    else:
        test_data = pickle.load(
            open(
                "med_exp/graph/results/explanations/hetero_gnn_explainer/test_graph.pkl",
                "rb",
            )
        ).to(device)

    # get predictions
    test_data = test_data.to(device)
    test_edges = test_data["notes", "links", "icds"].edge_label_index
    with torch.no_grad():
        model_preds = model(test_data)
        test_pred = model_preds > 0

    # load the edge masks
    filenum = 2500

    if noDupEdges:
        pickle_filename = f"med_exp/graph/results/explanations/hetero_gnn_explainer/NoDupEdges/gnnexp_gnn_hier_med_graph_3GATConv_pred_edge_to_comp_g_edge_mask_{filenum}.pkl"
    else:
        pickle_filename = f"med_exp/graph/results/explanations/hetero_gnn_explainer/gnnexp_gnn_hier_med_graph_3GATConv_pred_edge_to_comp_g_edge_mask_{filenum}.pkl"

    explainer_preds_2_neighbors, explainer_preds_3_neighbors, model_indices = [], [], []

    edge_masks = pickle.load(open(pickle_filename, "rb"))
    mask_dict_indices = list(edge_masks.keys())

    true_indices = []
    target_etype = ("notes", "links", "icds")

    for i in range(filenum + 1):
        if test_pred[i].item() is True:
            model_indices.append(i)
            true_indices.append(i)
            print(f"Calculating prediction value for edge idx {i}")
            src_nid, tgt_nid = test_edges[0][i], test_edges[1][i]
            src_ntype, tgt_ntype = "notes", "icds"

            mask = edge_masks[mask_dict_indices[len(true_indices) - 1]]
            mask = {etype: mask[etype].detach().sigmoid().to(device) for etype in mask}

            # get the k-hop subgraph
            (comp_g_src_nid, comp_g_tgt_nid, comp_g_k_hop, mapping) = (
                hetero_src_tgt_khop_in_subgraph(
                    src_ntype,
                    src_nid,
                    tgt_ntype,
                    tgt_nid,
                    test_data,
                    3,
                    device=device,
                )
            )

            # get the explanation for beam search 2 and beam search 3
            (
                comp_g_src_nid_2,
                comp_g_tgt_nid_2,
                comp_g_2,
                beam_search_mapping_2,
            ) = hetero_src_tgt_khop_in_subgraph(
                src_ntype,
                comp_g_src_nid,
                tgt_ntype,
                comp_g_tgt_nid,
                comp_g_k_hop,
                3,
                device=device,
                edge_weights=mask,
                num_neighbors=2,
            )

            comp_g_2[target_etype].edge_label_index = torch.tensor(
                [[comp_g_src_nid_2], [comp_g_tgt_nid_2]]
            )
            explainer_pred = model(comp_g_2)
            explainer_preds_2_neighbors.append(explainer_pred.item())

            (
                comp_g_src_nid_3,
                comp_g_tgt_nid_3,
                comp_g_3,
                beam_search_mapping_3,
            ) = hetero_src_tgt_khop_in_subgraph(
                src_ntype,
                comp_g_src_nid,
                tgt_ntype,
                comp_g_tgt_nid,
                comp_g_k_hop,
                3,
                device=device,
                edge_weights=mask,
                num_neighbors=3,
            )

            comp_g_3[target_etype].edge_label_index = torch.tensor(
                [[comp_g_src_nid_3], [comp_g_tgt_nid_3]]
            )
            explainer_pred = model(comp_g_3)
            explainer_preds_3_neighbors.append(explainer_pred.item())

    explainer_preds_2_neighbors = (
        torch.tensor(explainer_preds_2_neighbors).to(device).sigmoid()
    )
    explainer_preds_3_neighbors = (
        torch.tensor(explainer_preds_3_neighbors).to(device).sigmoid()
    )
    model_indices = torch.tensor(model_indices).to(device)
    model_preds = model_preds[model_indices].sigmoid()

    exp_preds_2 = explainer_preds_2_neighbors.cpu().detach()
    exp_preds_3 = explainer_preds_3_neighbors.cpu().detach()
    mod_preds = model_preds.cpu().detach()

    if noDupEdges:
        torch.save(
            exp_preds_2,
            "med_exp/graph/results/explanations/hetero_gnn_explainer/NoDupEdges/explainer_preds_2_neighbors.pt",
        )
        torch.save(
            exp_preds_3,
            "med_exp/graph/results/explanations/hetero_gnn_explainer/NoDupEdges/explainer_preds_3_neighbors.pt",
        )
        torch.save(
            model_preds,
            "med_exp/graph/results/explanations/hetero_gnn_explainer/NoDupEdges/model_preds.pt",
        )
    else:
        torch.save(
            exp_preds_2,
            "med_exp/graph/results/explanations/hetero_gnn_explainer/explainer_preds_2_neighbors.pt",
        )
        torch.save(
            exp_preds_3,
            "med_exp/graph/results/explanations/hetero_gnn_explainer/explainer_preds_3_neighbors.pt",
        )
        torch.save(
            model_preds,
            "med_exp/graph/results/explanations/hetero_gnn_explainer/model_preds.pt",
        )

    if noDupEdges:
        model_preds = (
            torch.load(
                "med_exp/graph/results/explanations/hetero_gnn_explainer/NoDupEdges/model_preds.pt"
            )
            .to(device)
            .detach()
        )
        exp_preds_2 = torch.load(
            "med_exp/graph/results/explanations/hetero_gnn_explainer/NoDupEdges/explainer_preds_2_neighbors.pt"
        ).to(device)
        exp_preds_3 = torch.load(
            "med_exp/graph/results/explanations/hetero_gnn_explainer/NoDupEdges/explainer_preds_3_neighbors.pt"
        ).to(device)
    else:
        model_preds = (
            torch.load(
                "med_exp/graph/results/explanations/hetero_gnn_explainer/model_preds.pt"
            )
            .to(device)
            .detach()
        )
        exp_preds_2 = torch.load(
            "med_exp/graph/results/explanations/hetero_gnn_explainer/explainer_preds_2_neighbors.pt"
        ).to(device)
        exp_preds_3 = torch.load(
            "med_exp/graph/results/explanations/hetero_gnn_explainer/explainer_preds_3_neighbors.pt"
        ).to(device)

    faithfulness_2 = (model_preds - exp_preds_2).abs().mean()
    faithfulness_3 = (model_preds - exp_preds_3).abs().mean()

    print(f"Faithfulness (2 neighbors): {faithfulness_2}")
    print(f"Faithfulness (3 neighbors): {faithfulness_3}")

    # convert to probability distributions
    model_dist = torch.stack([1 - model_preds, model_preds], dim=-1)
    exp_dist_2 = torch.stack([1 - exp_preds_2, exp_preds_2], dim=-1)
    exp_dist_3 = torch.stack([1 - exp_preds_3, exp_preds_3], dim=-1)

    # clamp and renormalize
    eps = 1e-8
    model_dist = torch.clamp(model_dist, min=eps, max=1 - eps)
    exp_dist_2 = torch.clamp(exp_dist_2, min=eps, max=1 - eps)
    exp_dist_3 = torch.clamp(exp_dist_3, min=eps, max=1 - eps)

    model_dist = model_dist / model_dist.sum(dim=-1, keepdim=True)
    exp_dist_2 = exp_dist_2 / exp_dist_2.sum(dim=-1, keepdim=True)
    exp_dist_3 = exp_dist_3 / exp_dist_3.sum(dim=-1, keepdim=True)

    kl_div_2 = F.kl_div(
        model_dist.log(),
        exp_dist_2,
        reduction="batchmean",
        log_target=False,
    )
    print(1 - torch.exp(-kl_div_2))

    kl_div_3 = F.kl_div(
        model_dist.log(),
        exp_dist_3,
        reduction="batchmean",
        log_target=False,
    )
    print(1 - torch.exp(-kl_div_3))
