import torch
import pickle, os
from pathlib import Path
import torch_geometric.transforms as T
from torch_geometric import seed_everything
from gnn_hier_graph import Model
from PaGELink import PaGELink

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
        model.load_state_dict(torch.load('results/hetero_gnn_model.pt'))
        model.eval()

        # initialize the explainer
        pagelink = PaGELink(
            model=model,
            src_ntype='cases',
            tgt_ntype='articles',
            num_epochs=20
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
        with torch.no_grad():
                test_pred = model(test_data) > 0

        # explain every edge  
        pickle.dump(test_data, open('results/explanations/pagelink/pagelink_graph.pkl', 'wb'))     
        # explain the following edges
        edges_to_explain = [5, 9, 12, 15, 16, 18, 20, 23, 432, 435, 442, 443, 445, 1936, 6497, 6581]
        count = 0
        pred_edge_to_comp_g_edge_mask = {}
        for i in range(test_edges.size(1)):
                if test_pred[i].item() is True:
                        if count in edges_to_explain:
                                print(f'Explaining edge idx {i}')
                                case_node, article_node = test_edges[0][i], test_edges[1][i]
                                comp_g_edge_mask_dict = pagelink.explain(
                                case_node, article_node, test_data, ('cases', 'violate', 'articles')
                                )
                                src_tgt = (('cases', case_node), ('articles', article_node))
                                pred_edge_to_comp_g_edge_mask[src_tgt] = comp_g_edge_mask_dict
                count += 1
                        
        # save explanations
        print('Saving explanations...')
        if not os.path.exists('results/explanations/pagelink/'):
                os.makedirs('results/explanations/pagelink/')
                
        saved_edge_explanation_file = f'pagelink_{model_name}_pred_edge_to_comp_g_edge_mask.pkl'   
        saved_edge_explanation_path = Path.cwd().joinpath('results/explanations/pagelink/', saved_edge_explanation_file)
        pickle.dump(pred_edge_to_comp_g_edge_mask, open(saved_edge_explanation_path, 'wb'))