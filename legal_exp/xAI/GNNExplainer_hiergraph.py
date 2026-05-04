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
        os.makedirs('./legal_exp/results/explanations/hetero_gnn_explainer/', exist_ok=True)
        pickle.dump(test_data, open('./legal_exp/results/explanations/hetero_gnn_explainer/graph.pkl', 'wb'))     
        pred_edge_to_comp_g_edge_mask = {}
        count = 0
        for i in range(test_edges.size(1)):
                if test_pred[i].item() is True and not bool(test_label[i].item()):
                        print(f'Explaining edge idx {i}')
                        count += 1
                        case_node, article_node = test_edges[0][i], test_edges[1][i]
                        
                        # case extraction
                        case_txt = getNodeText(case_node.item(), 'cases')
                        article_txt = getNodeText(article_node.item(), 'articles')
                        
                        os.makedirs('./datasets/cases/', exist_ok=True)
                        os.makedirs('./datasets/articles/', exist_ok=True)
                
                        # with open(f'./datasets/cases/{case_node.item()}.txt', 'w') as f:
                        #         f.write(case_txt)
                        # with open(f'./datasets/articles/{article_node.item()}.txt', 'w') as f:
                        #         f.write(article_txt)
                        # with open('./datasets/orig_to_exp_mapping.txt', 'a') as f:
                        #         f.write(f"Explanation {count} is (Case {case_node.item()}, Article {article_node.item()})\n")
                        
                        comp_g_edge_mask_dict = gnn_explainer.explain(case_node, article_node, test_data, ('cases', 'violate', 'articles'), device, num_hops=3)
                        src_tgt = (('cases', case_node), ('articles', article_node))
                        pred_edge_to_comp_g_edge_mask[src_tgt] = comp_g_edge_mask_dict
                        
        # save explanations
        print('Saving explanations...')
        if not os.path.exists('./legal_exp/results/explanations/hetero_gnn_explainer/'):
                os.makedirs('./legal_exp/results/explanations/hetero_gnn_explainer/')
                
        saved_edge_explanation_file = f'gnnexp_{model_name}_pred_edge_to_comp_g_edge_mask.pkl'   
        saved_edge_explanation_path = Path.cwd().joinpath('./legal_exp/results/explanations/hetero_gnn_explainer/', saved_edge_explanation_file)
        pickle.dump(pred_edge_to_comp_g_edge_mask, open(saved_edge_explanation_path, 'wb'))