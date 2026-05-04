import pickle
import os
import torch
from pathlib import Path
import torch_geometric.transforms as T
from torch_geometric import seed_everything
from med_exp.graph.gnn_med_graph import Model
from HeteroGNNExplainer import HeteroGNNExplainer
from med_exp.graph.med_graph_utils import getNodeText

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
    gnn_explainer = HeteroGNNExplainer(
        model=model, src_ntype="notes", tgt_ntype="icds", num_epochs=50
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
    with torch.no_grad():
        test_pred = model(test_data) > 0

    # (0, 2500), (2501, 4890), (4891, 6918), (6919, 9424)
    lower_limit = 0
    upper_limit = 2500

    # explain every edge
    if noDupEdges:
        pickle.dump(test_data, open('med_exp/graph/results/explanations/hetero_gnn_explainer/NoDupEdges/test_graph.pkl', 'wb'))
    else:
        pickle.dump(test_data, open('med_exp/graph/results/explanations/hetero_gnn_explainer/DupEdges/test_graph.pkl', 'wb'))

    pred_edge_to_comp_g_edge_mask = {}
    count, val = 0, 0
    for i in range(lower_limit, test_edges.size(1)):
        if test_pred[i].item() is True:
            print(f"Explaining edge idx {i}")
            count += 1
            note_node, icd_node = test_edges[0][i], test_edges[1][i]

            # note extraction
            note_txt = getNodeText(note_node.item(), "notes")
            icd_txt = getNodeText(icd_node.item(), "icds")

            os.makedirs("med_exp/data/notes/", exist_ok=True)
            os.makedirs("med_exp/data/icds/", exist_ok=True)

            # with open(f'med_exp/data/notes/{note_node.item()}.txt', 'w') as f:
            #         f.write(note_txt)
            # with open(f'med_exp/data/icds/{icd_node.item()}.txt', 'w') as f:
            #         f.write(icd_txt)
            # with open('med_exp/data/orig_to_exp_mapping.txt', 'a') as f:
            #         f.write(f"Explanation {count+3818} is (Note {note_node.item()}, ICD {icd_node.item()})\n")

            comp_g_edge_mask_dict = gnn_explainer.explain(
                note_node,
                icd_node,
                test_data,
                ("notes", "links", "icds"),
                device,
                num_hops=layers,
            )
            src_tgt = (("notes", note_node), ("icds", icd_node))
            pred_edge_to_comp_g_edge_mask[src_tgt] = comp_g_edge_mask_dict

        if i == upper_limit:
            val = i
            break

    # save explanations
    print("Saving explanations...")
    saved_edge_explanation_file = (
        f"gnnexp_{model_name}_pred_edge_to_comp_g_edge_mask_{val}.pkl"
    )

    if noDupEdges:
        if not os.path.exists("med_exp/graph/results/explanations/hetero_gnn_explainer/NoDupEdges/"):
            os.makedirs("med_exp/graph/results/explanations/hetero_gnn_explainer/NoDupEdges/")

        saved_edge_explanation_path = Path.cwd().joinpath(
            "med_exp/graph/results/explanations/hetero_gnn_explainer/NoDupEdges/",
            saved_edge_explanation_file,
        )
    else:
        if not os.path.exists("med_exp/graph/results/explanations/hetero_gnn_explainer/"):
            os.makedirs("med_exp/graph/results/explanations/hetero_gnn_explainer/")

        saved_edge_explanation_path = Path.cwd().joinpath(
            "med_exp/graph/results/explanations/hetero_gnn_explainer/",
            saved_edge_explanation_file,
        )
    pickle.dump(pred_edge_to_comp_g_edge_mask, open(saved_edge_explanation_path, "wb"))
