import torch
import pickle, os
from pathlib import Path
import torch_geometric.transforms as T
from torch_geometric import seed_everything
from legal_exp.graph.gnn_hier_graph import Model
from models.HeteroGNNExplainer import HeteroGNNExplainer
from utils.Graph_utils import getNodeText

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
seed_everything(4321)

if __name__ == "__main__":
        hier_graph = pickle.load(open('datasets/Hetero_Data_With_Self_Loops.pkl', 'rb'))
        case_embeds_legal_bert = pickle.load(open('../LegalGraph/code/dumps/legalbert_embeds_train.pkl', 'rb'))
        case_embeds_legal_bert = torch.cat(case_embeds_legal_bert, dim=0)
        hidden_dim = 64

        # retrieve the model
        model_name = 'gnn_hier_graph_3GATConv'
        model = Model(hier_graph, case_embeds_legal_bert, hier_graph['articles'].x.shape[1], hidden_dim)
        model.to(device)
        model.load_state_dict(torch.load('./legal_exp/results/hetero_gnn_model.pt'))
        model.eval()

        # initialize the explainer
        gnn_explainer = HeteroGNNExplainer(
                model=model,
                src_ntype='cases',
                tgt_ntype='articles',
                num_epochs=50
        ).to(device)
        
        # explain the test edges
        transform = T.RandomLinkSplit(
                    num_val=0.2,
                    num_test=0.2,  
                    disjoint_train_ratio=0.3,
                    neg_sampling_ratio=2.0,
                    # add_negative_train_samples=False,
                    edge_types=("cases", "violate", "articles"),
                    rev_edge_types=("articles", "rev_violate", "cases"))
        train_data, val_data, test_data = transform(hier_graph)
        del hier_graph, train_data, val_data, transform
        
        # get predictions
        test_data = test_data.to(device)
        test_edges = test_data['cases', 'violate', 'articles'].edge_label_index
        test_label = test_data["cases", "violate", "articles"].edge_label.to(device)
        with torch.no_grad():
                test_pred = model(test_data) > 0

        # explain every edge
        base_dir = Path.cwd().joinpath('./legal_exp/results/explanations/hetero_gnn_explainer/')
        os.makedirs(base_dir, exist_ok=True)
        # pickle.dump(test_data, open(base_dir.joinpath('graph.pkl'), 'wb'))     

        # Categorize test edges
        categories = {
                'tp': [],
                'tn': [],
                'fp': [],
                'fn': []
        }
        for i in range(test_edges.size(1)):
                pred = bool(test_pred[i].item())
                label = bool(test_label[i].item())
                if pred and label:
                        categories['tp'].append(i)
                elif not pred and not label:
                        categories['tn'].append(i)
                elif pred and not label:
                        categories['fp'].append(i)
                elif not pred and label:
                        categories['fn'].append(i)

        for pred_type, indices in categories.items():
                if pred_type in ['tp']: # no need to process
                        continue
                print(f"Processing category: {pred_type} (count: {len(indices)})")
                pred_edge_to_comp_g_edge_mask = {}
                count = 0
                
                # Create subfolder for this category
                subfolder_dir = base_dir.joinpath(pred_type)
                os.makedirs(subfolder_dir, exist_ok=True)

                # Symlink graph.pkl from parent directory to the subfolder
                subfolder_graph_path = subfolder_dir.joinpath('graph.pkl')
                if not os.path.exists(subfolder_graph_path):
                        # Relative symlink
                        os.symlink('../graph.pkl', subfolder_graph_path)

                for idx in indices:
                        print(f'Explaining edge idx {idx} ({pred_type})')
                        count += 1
                        case_node, article_node = test_edges[0][idx], test_edges[1][idx]
                        
                        comp_g_edge_mask_dict = gnn_explainer.explain(case_node, article_node, test_data, ('cases', 'violate', 'articles'), device, num_hops=3)
                        src_tgt = (('cases', case_node), ('articles', article_node))
                        pred_edge_to_comp_g_edge_mask[src_tgt] = comp_g_edge_mask_dict

                        if count > 500: # process only 500 per category to save memory
                                break

                # save explanations for this category
                print(f'Saving {pred_type} explanations...')
                saved_edge_explanation_file = f'gnnexp_{model_name}_pred_edge_to_comp_g_edge_mask.pkl'
                saved_edge_explanation_path = subfolder_dir.joinpath(saved_edge_explanation_file)
                pickle.dump(pred_edge_to_comp_g_edge_mask, open(saved_edge_explanation_path, 'wb'))