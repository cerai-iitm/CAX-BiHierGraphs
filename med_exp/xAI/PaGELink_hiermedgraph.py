import torch
import pickle
import os
import datetime
from pathlib import Path
import torch_geometric.transforms as T
from torch_geometric import seed_everything
from med_exp.graph.gnn_med_graph import Model
from models.PaGELink import PaGELink

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
seed_everything(4321)

print("CUDA device count:", torch.cuda.device_count())
print("Visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Current device index:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))

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
            torch.load("med_exp/graph/results/NoDupEdges/hetero_gnn_model_neg_10_NoDupEdges.pt")
        )
    else:
        model.load_state_dict(
            torch.load("med_exp/graph/results/DupEdges/hetero_gnn_model_neg_10.pt")
        )
    model.eval()

    # initialize the explainer
    pagelink = PaGELink(
        model=model,
        src_ntype="notes",
        tgt_ntype="icds",
        num_epochs=20,
    ).to(device)

    # explain the test edges
    transform = T.RandomLinkSplit(
        num_val=0.2,
        num_test=0.2,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=10.0,
        # add_negative_train_samples=False,
        edge_types=("notes", "links", "icds"),
        rev_edge_types=("icds", "rev_links", "notes"),
    )
    train_data, val_data, test_data = transform(hier_graph)
    del hier_graph, train_data, val_data, transform

    # get predictions
    test_data = test_data.to(device)
    test_edges = test_data["notes", "links", "icds"].edge_label_index
    test_label = test_data["notes", "links", "icds"].edge_label.to(device)
    with torch.no_grad():
        test_pred = model(test_data) > 0

    # Ensure output directory exists and save the test graph for downstream processing
    base_dir = Path.cwd().joinpath(
        "med_exp/graph/results/explanations/hetero_pagelink",
        "NoDupEdges" if noDupEdges else "DupEdges",
    )
    os.makedirs(base_dir, exist_ok=True)
    pickle.dump(test_data, open(base_dir.joinpath("graph.pkl"), "wb"))

    # Categorize test edges
    categories = {"tp": [], "tn": [], "fp": [], "fn": []}
    for i in range(test_edges.size(1)):
        pred = bool(test_pred[i].item())
        label = bool(test_label[i].item())
        if pred and label:
            categories["tp"].append(i)
        elif not pred and not label:
            categories["tn"].append(i)
        elif pred and not label:
            categories["fp"].append(i)
        elif not pred and label:
            categories["fn"].append(i)

    for pred_type, indices in categories.items():
        print(f"Processing category: {pred_type} (count: {len(indices)})")
        pred_edge_to_paths = {}
        count = 0

        # Create subfolder for this category
        subfolder_dir = base_dir.joinpath(pred_type)
        os.makedirs(subfolder_dir, exist_ok=True)

        # Symlink graph.pkl from parent directory to the subfolder
        subfolder_graph_path = subfolder_dir.joinpath("graph.pkl")
        if not subfolder_graph_path.exists():
            os.symlink("../graph.pkl", subfolder_graph_path)

        for idx in indices:
            print(f"{datetime.datetime.now()} Explaining edge idx {idx} ({pred_type})")
            note_node, icd_node = test_edges[0][idx], test_edges[1][idx]

            comp_g_paths = pagelink.explain(
                note_node,
                icd_node,
                test_data,
                ("notes", "links", "icds"),
                num_hops=layers,
            )
            src_tgt = (("notes", note_node), ("icds", icd_node))
            pred_edge_to_paths[src_tgt] = comp_g_paths

            count += 1
            if count > 500:  # process only 500 per category to save memory
                break

        # save explanations for this category
        print(f"Saving {pred_type} explanations...")
        saved_edge_explanation_file = f"pagelink_{model_name}_pred_edge_to_comp_g_edge_mask_{pred_type}.pkl"
        saved_edge_explanation_path = subfolder_dir.joinpath(saved_edge_explanation_file)
        pickle.dump(pred_edge_to_paths, open(saved_edge_explanation_path, "wb"))