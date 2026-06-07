import sys
import pickle, os
from pathlib import Path
import torch
import torch_geometric.transforms as T
from torch_geometric import seed_everything
from legal_exp.graph.gnn_hier_graph import Model
from models.HeteroPGExplainer import HeteroPGExplainer
import pickle as pkl
import gc

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device:", device)

FLUSH_EVERY   = 100 # save partial results every N edges to bound RAM
MAX_PER_CAT   = 500 # hard cap per category

if __name__ == "__main__":
    seed_everything(4321)

    hier_graph = pickle.load(open('datasets/Hetero_Data_With_Self_Loops.pkl', 'rb'))
    case_embeds_legal_bert = pickle.load(
        open('../LegalGraph/code/dumps/legalbert_embeds_train.pkl', 'rb')
    )
    case_embeds_legal_bert = torch.cat(case_embeds_legal_bert, dim=0)
    hidden_dim = 64

    model_name = 'gnn_hier_graph_3GATConv'
    model = Model(
        hier_graph,
        case_embeds_legal_bert,
        hier_graph['articles'].x.shape[1],
        hidden_dim,
    )

    del case_embeds_legal_bert
    gc.collect()

    model.to(device)
    model.load_state_dict(
        torch.load('./legal_exp/results/hetero_gnn_model.pt', map_location=device)
    )
    model.eval()

    test_data = pickle.load(
        open('./legal_exp/results/explanations/hetero_gnn_explainer/graph.pkl', 'rb')
    )
    test_data = test_data.to(device)

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
    ).to(device)

    explainer_model_save_path = (
        './legal_exp/results/explanations/hetero_pg_explainer/hetero_pg_explainer.pt'
    )
    results_path = './legal_exp/results/explanations/hetero_pg_explainer'
    os.makedirs(results_path, exist_ok=True)

    Task = input("\nEnter t for training model \nEnter e for explaining\n")

    if Task == 't':
        print("\nTraining\n")
        pgexplainer.train_mask_generator(
            test_data, ("cases", "violate", "articles"), device, batch_size=72
        )
        torch.save(pgexplainer.state_dict(), explainer_model_save_path)
        with open(os.path.join(results_path, 'epochwise_losses.pkl'), 'wb') as file:
            pkl.dump(pgexplainer.epoch_loss, file)

    elif Task == 'e':
        print("\nExplaining\n")
        pgexplainer.load_state_dict(
            torch.load(explainer_model_save_path, map_location=device)
        )
        pgexplainer.eval()

        edge_label_index = test_data['cases', 'violate', 'articles'].edge_label_index
        test_label = test_data["cases", "violate", "articles"].edge_label.to(device)

        with torch.no_grad():
            test_pred = model(test_data) > 0

        preds  = test_pred.view(-1).cpu().bool()
        labels = test_label.view(-1).cpu().bool()

        categories = {
            'tp': torch.where( preds &  labels)[0].tolist(),
            'tn': torch.where(~preds & ~labels)[0].tolist(),
            'fp': torch.where( preds & ~labels)[0].tolist(),
            'fn': torch.where(~preds &  labels)[0].tolist(),
        }

        del test_pred, test_label
        torch.cuda.empty_cache()

        # pull edge index to CPU once — avoids repeated GPU→CPU copies
        # and keeps the tensor alive without holding device memory needlessly
        test_edges = edge_label_index.cpu()
        del edge_label_index

        base_dir = Path(results_path)

        base_graph_path = base_dir / 'graph.pkl'
        if not os.path.lexists(base_graph_path):
            os.symlink('../hetero_gnn_explainer/graph.pkl', base_graph_path)

        for pred_type, indices in categories.items():
            indices = indices[:MAX_PER_CAT]
            print(f"Processing category: {pred_type} (count: {len(indices)})")

            subfolder_dir = base_dir / pred_type
            os.makedirs(subfolder_dir, exist_ok=True)

            # same lexists fix for per-category symlink
            subfolder_graph_path = subfolder_dir / 'graph.pkl'
            if not os.path.lexists(subfolder_graph_path):
                os.symlink('../graph.pkl', subfolder_graph_path)

            saved_edge_explanation_file = (
                f'pgexp_{model_name}_pred_edge_to_comp_g_edge_mask.pkl'
            )
            saved_edge_explanation_path = subfolder_dir / saved_edge_explanation_file

            # Load any previously saved chunk so we can append to it
            if saved_edge_explanation_path.exists():
                with open(saved_edge_explanation_path, 'rb') as f:
                    pred_edge_to_comp_g_edge_mask = pickle.load(f)
            else:
                pred_edge_to_comp_g_edge_mask = {}

            for i, idx in enumerate(indices):
                case_node_id    = test_edges[0][idx].item()
                article_node_id = test_edges[1][idx].item()

                print(f'[{pred_type}] Explaining edge {i+1}/{len(indices)} (idx={idx})')

                # no_grad prevents accumulation of computation graphs
                # across hundreds of explain() calls — critical for GPU memory
                with torch.no_grad():
                    comp_g_edge_mask_dict = pgexplainer.explain(
                        test_edges[0][idx].to(device),
                        test_edges[1][idx].to(device),
                        test_data,
                        ('cases', 'violate', 'articles'),
                        device,
                        num_hops=3,
                    )

                # Detach and move masks to CPU immediately
                if isinstance(comp_g_edge_mask_dict, dict):
                    comp_g_edge_mask_dict = {
                        k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in comp_g_edge_mask_dict.items()
                    }
                elif isinstance(comp_g_edge_mask_dict, torch.Tensor):
                    comp_g_edge_mask_dict = comp_g_edge_mask_dict.detach().cpu()

                src_tgt = (('cases', case_node_id), ('articles', article_node_id))
                pred_edge_to_comp_g_edge_mask[src_tgt] = comp_g_edge_mask_dict

                # flush to disk periodically so RAM doesn't balloon
                # if a category has hundreds of edges
                if (i + 1) % FLUSH_EVERY == 0:
                    print(f'  Flushing {len(pred_edge_to_comp_g_edge_mask)} entries...')
                    with open(saved_edge_explanation_path, 'wb') as f:
                        pickle.dump(pred_edge_to_comp_g_edge_mask, f)

            # Final save for this category
            print(f'Saving {pred_type} explanations ({len(pred_edge_to_comp_g_edge_mask)} total)...')
            with open(saved_edge_explanation_path, 'wb') as f:
                pickle.dump(pred_edge_to_comp_g_edge_mask, f)

            del pred_edge_to_comp_g_edge_mask
            gc.collect()
            torch.cuda.empty_cache()