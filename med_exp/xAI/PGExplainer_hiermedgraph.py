import sys
import pickle, os
from pathlib import Path
import torch
import torch_geometric.transforms as T
from torch_geometric import seed_everything
from med_exp.graph.gnn_med_graph import Model
from models.HeteroPGExplainer import HeteroPGExplainer
import gc

# Device selection
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device:", device)

# Configuration constants
FLUSH_EVERY = 100  # Save partial results every N edges to bound RAM
MAX_PER_CAT = 500   # Hard cap per category

if __name__ == "__main__":
    # Seed for reproducibility
    seed_everything(4321)

    # ---------------------------------------------------------------------
    # Load hierarchical medical graph (choose NoDupEdges version by default)
    # ---------------------------------------------------------------------
    graph_path = "med_exp/graph/MedGraph_NoDupEdges.pkl"
    hier_graph = pickle.load(open(graph_path, "rb"))

    # ---------------------------------------------------------------------
    # Model hyper‑parameters (mirroring GNNExplainer script)
    # ---------------------------------------------------------------------
    hidden_dim = 64
    layers = 3
    model_name = f"gnn_hier_med_graph_{layers}GATConv"

    # Initialise the GNN model for the medical graph
    model = Model(
        hier_graph,
        hier_graph["notes"].x.shape[1],
        hidden_dim,
        layers,
    )
    model.to(device)

    # Load the pre‑trained checkpoint (NoDupEdges version) with tolerant loading
    checkpoint_path = "med_exp/graph/results/NoDupEdges/hetero_gnn_model_neg_10_NoDupEdges.pt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warning] Missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"[Warning] Unexpected keys when loading checkpoint: {unexpected}")
    model.eval()

    # ---------------------------------------------------------------------
    # Prepare test data (same split logic as GNNExplainer script)
    # ---------------------------------------------------------------------
    transform = T.RandomLinkSplit(
        num_val=0.2,
        num_test=0.2,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=10.0,
        edge_types=("notes", "links", "icds"),
        rev_edge_types=("icds", "rev_links", "notes"),
    )
    train_data, val_data, test_data = transform(hier_graph)
    del hier_graph, train_data, val_data, transform
    test_data = test_data.to(device)
    test_edges = test_data["notes", "links", "icds"].edge_label_index

    # ---------------------------------------------------------------------
    # Initialise PGExplainer for the medical graph
    # ---------------------------------------------------------------------
    pgexplainer = HeteroPGExplainer(
        model=model,
        num_hops=3,
        ghetero=test_data,
        lr=0.005,
        alpha1=1e-2,
        alpha2=5e-4,
        in_dim=64,
        K=2,
        mask_generator_hidden_dim=64,
        num_epochs=100,
        device=device,
        src_ntype='notes',
        tgt_ntype='icds',
    ).to(device)

    # ---------------------------------------------------------------------
    # Output directories
    # ---------------------------------------------------------------------
    results_path = "med_exp/graph/results/explanations/hetero_pg_explainer/NoDupEdges"
    os.makedirs(results_path, exist_ok=True)
    explainer_model_save_path = os.path.join(results_path, "hetero_pg_explainer.pt")

    # ---------------------------------------------------------------------
    # Interactive mode selection
    # ---------------------------------------------------------------------
    Task = input("\nEnter t for training model \nEnter e for explaining\n")

    if Task == 't':
        print("\nTraining PGExplainer\n")
        pgexplainer.train_mask_generator(
            test_data, ("notes", "links", "icds"), device, batch_size=72
        )
        torch.save(pgexplainer.state_dict(), explainer_model_save_path)
        # Save epoch‑wise loss for later analysis
        epoch_loss_path = os.path.join(results_path, "epochwise_losses.pkl")
        with open(epoch_loss_path, 'wb') as f:
            pickle.dump(pgexplainer.epoch_loss, f)

    elif Task == 'e':
        print("\nExplaining with PGExplainer\n")
        pgexplainer.load_state_dict(torch.load(explainer_model_save_path, map_location=device))
        pgexplainer.eval()

        # Compute predictions to segregate edges into categories
        with torch.no_grad():
            test_pred = model(test_data) > 0
        preds = test_pred.view(-1).cpu().bool()
        labels = test_data["notes", "links", "icds"].edge_label.view(-1).cpu().bool()

        categories = {
            'tp': torch.where(preds & labels)[0].tolist(),
            'tn': torch.where(~preds & ~labels)[0].tolist(),
            'fp': torch.where(preds & ~labels)[0].tolist(),
            'fn': torch.where(~preds & labels)[0].tolist(),
        }

        # Release GPU memory before the per‑category loop
        del test_pred, test_data["notes", "links", "icds"].edge_label
        torch.cuda.empty_cache()

        base_dir = Path(results_path)
        base_graph_path = base_dir / 'graph.pkl'
        if not base_graph_path.exists():
            # Symlink to the original test graph for convenience
            os.symlink('../hetero_gnn_explainer/NoDupEdges/graph.pkl', str(base_graph_path))

        edge_index_cpu = test_edges.cpu()
        del test_edges

        for pred_type, indices in categories.items():
            # Limit per‑category size to keep memory bounded
            indices = indices[:MAX_PER_CAT]
            print(f"Processing category: {pred_type} (count: {len(indices)})")

            subfolder_dir = base_dir / pred_type
            os.makedirs(subfolder_dir, exist_ok=True)
            subgraph_path = subfolder_dir / 'graph.pkl'
            if not subgraph_path.exists():
                os.symlink('../graph.pkl', str(subgraph_path))

            saved_edge_explanation_file = f'pgexp_{model_name}_pred_edge_to_comp_g_edge_mask_{pred_type}.pkl'
            saved_edge_explanation_path = subfolder_dir / saved_edge_explanation_file

            # Load any existing partial results
            if saved_edge_explanation_path.exists():
                with open(saved_edge_explanation_path, 'rb') as f:
                    pred_edge_to_comp_g_edge_mask = pickle.load(f)
            else:
                pred_edge_to_comp_g_edge_mask = {}

            for i, idx in enumerate(indices):
                note_node_id = edge_index_cpu[0][idx].item()
                icd_node_id = edge_index_cpu[1][idx].item()
                print(f'[{pred_type}] Explaining edge {i+1}/{len(indices)} (idx={idx})')

                with torch.no_grad():
                    comp_g_edge_mask_dict = pgexplainer.explain(
                        edge_index_cpu[0][idx].to(device),
                        edge_index_cpu[1][idx].to(device),
                        test_data,
                        ("notes", "links", "icds"),
                        device,
                        num_hops=layers,
                    )

                # Detach tensors to CPU immediately
                if isinstance(comp_g_edge_mask_dict, dict):
                    comp_g_edge_mask_dict = {
                        k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in comp_g_edge_mask_dict.items()
                    }
                elif isinstance(comp_g_edge_mask_dict, torch.Tensor):
                    comp_g_edge_mask_dict = comp_g_edge_mask_dict.detach().cpu()

                src_tgt = (("notes", note_node_id), ("icds", icd_node_id))
                pred_edge_to_comp_g_edge_mask[src_tgt] = comp_g_edge_mask_dict

                # Periodic flush to disk
                if (i + 1) % FLUSH_EVERY == 0:
                    print(f'  Flushing {len(pred_edge_to_comp_g_edge_mask)} entries...')
                    with open(saved_edge_explanation_path, 'wb') as f:
                        pickle.dump(pred_edge_to_comp_g_edge_mask, f)

            # Final save for this category
            print(f'Saving {pred_type} explanations ({len(pred_edge_to_comp_g_edge_mask)} total)...')
            with open(saved_edge_explanation_path, 'wb') as f:
                pickle.dump(pred_edge_to_comp_g_edge_mask, f)

            # Cleanup for next category
            del pred_edge_to_comp_g_edge_mask
            gc.collect()
            torch.cuda.empty_cache()

    else:
        print("Invalid option. Exiting.")
