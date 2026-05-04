import os
import json
import pickle as pkl
import numpy as np
from collections import Counter
from tqdm import tqdm
import argparse
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, GraphConv, SAGEConv, to_hetero
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric import seed_everything

from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import matplotlib.pyplot as plt

class HierGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # self.conv1 = GATConv(input_dim, hidden_dim)
        # self.conv2 = GATConv(input_dim, hidden_dim)
        self.conv1 = GATConv(input_dim, hidden_dim, add_self_loops=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.conv3 = GATConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.activation = torch.nn.ReLU()

    def forward(self, x, edge_index):
        '''
        x: (num_nodes, input_dim)
        edge_index: (2, num_edges)

        Returns:
        x_out: (num_nodes, hidden_dim)
        '''
        print('GAT ')
        x_out = self.conv1(x, edge_index)
        x_out = self.activation(x_out)
        x_out = self.conv2(x_out, edge_index)
        x_out = self.activation(x_out)
        x_out = self.conv3(x_out, edge_index)
        return x_out

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_case, x_article, edge_label_index):
        # Convert node embeddings to edge-level representations:
        edge_feat_case = x_case[edge_label_index[0]]
        edge_feat_article = x_article[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_case * edge_feat_article).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, graph, case_attr, input_dim, hidden_dim):
        super().__init__()
        self.case_embed = torch.nn.Parameter(case_attr)
        # Set the embedding weights to pre-computed values
        #self.case_embed.weight.data.copy_(case_attr)

        # Instantiate homogeneous GNN:
        self.gnn = HierGNN(input_dim, hidden_dim)

        # Convert GNN model into a heterogeneous variant:
        #print(graph.metadata())
        self.gnn = to_hetero(self.gnn, metadata=graph.metadata())
        #print('Hetero GNN:\n', self.gnn)
        self.classifier = Classifier()

    def forward(self, graph, eweights=None):
        x_dict = {
          "articles": graph["articles"].x,
          "terms": graph["terms"].x,
          "facts": graph["facts"].x,
          "cases": self.case_embed,
        }
        
        if eweights:
            for etype in graph.edge_types:
                graph[etype].edge_weight = eweights[etype]
        else:
            for etype in graph.edge_types:
                graph[etype].edge_weight = torch.ones(graph[etype].edge_index.size(1),
                                                      device=graph[etype].edge_index.device)
        
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, graph.edge_index_dict)

        pred = self.classifier(
            x_dict["cases"],
            x_dict["articles"],
            graph["cases", "violate", "articles"].edge_label_index,
        )
        return pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help = '''Path to the saved hierarchical graph''', 
                        default = './datasets/Hetero_Data_With_Self_Loops.pkl', type = str)
    parser.add_argument('--case_embeds_path', help = '''Path to the saved case embeddings''', 
                        default = '/home/gokul/LegalGraph/code/dumps/legalbert_embeds_train.pkl', type = str)      
    parser.add_argument('--results_path', help = '''Path to save the results''',
                        default = './legal_exp/results', type = str)
    parser.add_argument('--model_save_path', help = '''Path to save the trained model''',
                        default = './legal_exp/results/hetero_gnn_model_neg_10.pt', type = str)
    parser.add_argument('--hidden_dim', help = '''Hidden dimension of the GNN''', 
                        default = 64, type = int)
    parser.add_argument('--lr', help = '''Learning rate for the optimzer''', 
                        default = 5e-4, type = float)
    parser.add_argument('--num_epochs', help = '''Number of epochs''', 
                        default = 500, type = int)
    parser.add_argument('--neg_sampling_ratio', help = '''Number of negative pairs''', 
                        default = 10.0, type = float)
    parser.add_argument('--rand_seed', help = 'Random seed for initializing and training the model', 
                        default = 4321, type = int) 
    parser.add_argument('--device', help = 'Device to be used for model training', 
                        default = 'cuda', choices = ['cuda', 'cpu'], type = str) 
    args = parser.parse_args()
    torch_device = torch.device('cuda') if (torch.cuda.is_available() and args.device == 'cuda') \
        else torch.device('cpu') 
    print(torch_device)
    seed_everything(args.rand_seed)
    
    #################################
    # Read and preprocess the graph
    #################################
    with open(args.dataset_path, 'rb') as file:
        hier_graph = pkl.load(file)
    for node_type in hier_graph.node_types:
        print('{}: {}'.format(node_type, hier_graph[node_type].x.shape)) 
    # for node_type in hier_graph.node_types:
    #     hier_graph[node_type, 'self_edge_{}'.format(node_type), node_type] = \
    #         torch.stack([torch.arange(hier_graph[node_type].num_nodes),
    #                      torch.arange(hier_graph[node_type].num_nodes)], dim = 0)
    
    # hier_graph = T.AddSelfLoops()(hier_graph)
    #print(hier_graph)
    #print(hier_graph.edge_index_dict)
    
    with open(args.case_embeds_path, 'rb') as file:
        case_embeds_legalbert = pkl.load(file)
    #print(type(case_embeds_legalbert), len(case_embeds_legalbert))
    case_embeds_legalbert = torch.cat(case_embeds_legalbert, dim=0)
    
    transform = T.RandomLinkSplit(
                    num_val=0.2,
                    num_test=0.2,  
                    disjoint_train_ratio=0.3,
                    neg_sampling_ratio=args.neg_sampling_ratio,
                    # add_negative_train_samples=False, 
                    edge_types=("cases", "violate", "articles"),
                    rev_edge_types=("articles", "rev_violate", "cases"),)
    train_data, val_data, test_data = transform(hier_graph)

    # Get all unique case node indices that participate in "violate" edges
    # case_indices = hier_graph["cases", "violate", "articles"].edge_index[0].unique()
    # n_cases = len(case_indices)

    # # Shuffle and split case indices (not edges)
    # perm = torch.randperm(n_cases)
    # n_val  = int(0.2 * n_cases)
    # n_test = int(0.2 * n_cases)

    # val_cases  = case_indices[perm[:n_val]]
    # test_cases = case_indices[perm[n_val:n_val + n_test]]
    # train_cases = case_indices[perm[n_val + n_test:]]

    # # Create edge masks based on which case the edge belongs to
    # edge_case_index = hier_graph["cases", "violate", "articles"].edge_index[0]

    # train_mask = torch.isin(edge_case_index, train_cases)
    # val_mask   = torch.isin(edge_case_index, val_cases)
    # test_mask  = torch.isin(edge_case_index, test_cases)

    # def build_split(hier_graph, edge_mask, all_train_edges):
    #     """
    #     hier_graph   : original full graph
    #     edge_mask    : boolean mask selecting edges for this split's supervision
    #     all_train_edges : edge_index of ALL training edges (used as message-passing
    #                     edges for val/test so the GNN has structural context)
    #     """
    #     data = copy.deepcopy(hier_graph)

    #     # ── Supervision edges (what we evaluate on) ──────────────────────────────
    #     full_edge_index = hier_graph["cases", "violate", "articles"].edge_index
    #     data["cases", "violate", "articles"].edge_label_index = full_edge_index[:, edge_mask]
    #     data["cases", "violate", "articles"].edge_label = torch.ones(edge_mask.sum(), dtype=torch.float)

    #     # ── Message-passing edges (structural context for the GNN) ───────────────
    #     # For val/test: only use training edges for message passing so there's
    #     # no leakage of val/test supervision edges into the GNN's aggregation.
    #     data["cases", "violate", "articles"].edge_index = all_train_edges

    #     return data

    # train_edge_index = hier_graph["cases", "violate", "articles"].edge_index[:, train_mask]

    # train_data = build_split(hier_graph, train_mask, train_edge_index)
    # val_data   = build_split(hier_graph, val_mask,   train_edge_index)
    # test_data  = build_split(hier_graph, test_mask,  train_edge_index)

    # num_articles = hier_graph["articles"].num_nodes
    # full_edge_index = hier_graph["cases", "violate", "articles"].edge_index
    # pos_set = set(zip(full_edge_index[0].tolist(), full_edge_index[1].tolist()))

    # def get_all_edges_with_labels(case_ids, num_articles, pos_set):
    #     all_edges, all_labels = [], []
    #     for case_id in case_ids.tolist():
    #         for article_id in range(num_articles):
    #             all_edges.append((case_id, article_id))
    #             all_labels.append(1.0 if (case_id, article_id) in pos_set else 0.0)
    #     return (torch.tensor(all_edges, dtype=torch.long).t(),
    #             torch.tensor(all_labels, dtype=torch.float))

    # train_data["cases", "violate", "articles"].edge_label_index, \
    # train_data["cases", "violate", "articles"].edge_label = get_all_edges_with_labels(train_cases, num_articles, pos_set)

    # val_data["cases", "violate", "articles"].edge_label_index, \
    # val_data["cases", "violate", "articles"].edge_label = get_all_edges_with_labels(val_cases, num_articles, pos_set)

    # print('Initializing link neighbor loader')
    # train_loader = LinkNeighborLoader(
    #     data=train_data,  
    #     num_neighbors=[20, 10],  
    #     neg_sampling_ratio=2.0,
    #     edge_label_index=(("cases", "violate", "articles"), edge_label_index),
    #     edge_label=edge_label,
    #     batch_size=128,
    #     shuffle=True)

    #################################
    # Initialize and train the model
    #################################
    print('Initializing the model')
    model = Model(hier_graph, case_embeds_legalbert, hier_graph['articles'].x.shape[1], args.hidden_dim)
    model.to(torch_device)
    model.train()
    #print(model)
    
    train_data = train_data.to(torch_device)
    train_edge_label_index = train_data["cases", "violate", "articles"].edge_label_index.to(torch_device)
    train_edge_label = train_data["cases", "violate", "articles"].edge_label.to(torch_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epochwise_losses = []
    # for epoch in range(1, args.num_epochs+1):
    #     total_loss = total_examples = 0
    #     for sampled_data in tqdm.tqdm(train_loader):
    #         optimizer.zero_grad()
    #         sampled_data = sampled_data.to(torch_device)
    #         cls_pred = model(sampled_data)
    #         loss = F.binary_cross_entropy_with_logits(cls_pred, sampled_data["cases", "violate", "articles"]['edge_label'])
    #         # Move `sampled_data` to the respective `device`
    #         # Run `forward` pass of the model
    #         # Apply binary cross entropy via
    #         # `F.binary_cross_entropy_with_logits(pred, ground_truth)`
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += float(loss) * cls_pred.numel()
    #         total_examples += cls_pred.numel()
    #     print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
    #     epochwise_losses.append(total_loss / total_examples)
    print(train_data)

    progress_bar = tqdm(range(1, args.num_epochs+1))
    for epoch in progress_bar:
        optimizer.zero_grad()
        cls_pred = model(train_data)
        loss = F.binary_cross_entropy_with_logits(cls_pred, train_edge_label)
        # Move `sampled_data` to the respective `device`
        # Run `forward` pass of the model
        # Apply binary cross entropy via
        # `F.binary_cross_entropy_with_logits(pred, ground_truth)`
        loss.backward()
        optimizer.step()
        #print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
        epochwise_losses.append(loss)
        # Update the progress bar description
        progress_bar.set_description(f"Epoch {epoch}/{args.num_epochs} Loss: {loss:.4f}")

    #print(epochwise_losses)
    with open(os.path.join(args.results_path, 'epochwise_losses.pkl'), 'wb') as file:
        pkl.dump(epochwise_losses, file)
    
    torch.save(model.state_dict(), args.model_save_path)
    #model = Model(hier_graph, case_embeds_legalbert, hier_graph['articles'].x.shape[1], args.hidden_dim)
    #model.load_state_dict(torch.load(args.model_save_path))
    #model.to(torch_device)
    
    # Running inference on the training data
    model.eval()
    with torch.no_grad():
        train_logit = model(train_data)
        train_pred = (train_logit> 0) # Converting logits to binary labels
    train_logit = train_logit.detach().cpu().numpy()
    train_pred = train_pred.detach().cpu().numpy()
    train_edge_label = train_edge_label.detach().cpu().numpy()
    print('============================================================')
    print(train_pred.shape, train_edge_label.shape)
    print(train_pred)
    print(train_edge_label)
    print(f"\nTrain AUC: {roc_auc_score(train_edge_label, train_logit):.4f}")
    print("Train confusion matrix =\n", confusion_matrix(train_edge_label, train_pred))
    print('============================================================')

    #############################################
    # Running inference on the validation edges
    #############################################
    val_data = val_data.to(torch_device)
    val_edge_label_index = val_data["cases", "violate", "articles"].edge_label_index
    val_edge_label = val_data["cases", "violate", "articles"].edge_label

    # val_loader = LinkNeighborLoader(
    #     data=val_data,
    #     num_neighbors=[20, 10],
    #     edge_label_index=(("cases", "violate", "articles"), edge_label_index),
    #     edge_label=edge_label,
    #     batch_size=3 * 128,
    #     shuffle=False)

    # preds = []
    # ground_truths = []
    # for sampled_data in tqdm(val_loader):
    #     with torch.no_grad():
    #         sampled_data = sampled_data.to(torch_device)
    #         cls_pred = model(sampled_data).round()
    #         preds.append(cls_pred)
    #         ground_truths.append(sampled_data["cases", "violate", "articles"]['edge_label'])
    #         # Collect predictions and ground-truths and write 
    #         # them into `preds` and `ground_truths`
    # pred = torch.cat(preds, dim=0).cpu().numpy()
    # ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    print('============================================================')
    print(val_data)
    model.eval()
    with torch.no_grad():
        val_logit = model(val_data)
        val_pred = (val_logit > 0) # Converting logits to binary labels
    val_logit = val_logit.detach().cpu().numpy()
    val_pred = val_pred.detach().cpu().numpy()
    # val_pred = (val_pred > 0) 
    val_edge_label = val_edge_label.detach().cpu().numpy()
    print(val_pred.shape, val_edge_label.shape)
    print(val_pred)
    print(val_edge_label)
    print(f"\nValidation AUC: {roc_auc_score(val_edge_label, val_logit):.4f}")
    print(f"\nF1 Score: {f1_score(val_edge_label, val_pred):.4f}")
    print("Validation confusion matrix =\n", confusion_matrix(val_edge_label, val_pred))
    print('============================================================')

    with open(os.path.join(args.results_path, 'val_predictions.pkl'), 'wb') as file:
        pkl.dump({'ground_truth': val_edge_label,
                  'pred': val_logit}, file)    
    with open(os.path.join(args.results_path, 'train_predictions.pkl'), 'wb') as file:
        pkl.dump({'ground_truth': train_edge_label,
                  'pred': train_logit}, file) 


    # case_ids = val_data["cases", "violate", "articles"].edge_label_index[0].cpu().numpy()
    # results = {}
    # for case_id in np.unique(case_ids):
    #     mask = (case_ids == case_id)
    #     results[case_id] = {
    #         'pred':  val_pred[mask],   # predicted violations for this case
    #         'label': val_edge_label[mask],  # ground truth violations for this case
    #     }
    # per_case_f1 = [
    #     f1_score(v['label'], v['pred'], average='binary', zero_division=0)
    #     for v in results.values()
    #     # if len(np.unique(v['label'])) > 1  # skip degenerate cases
    # ]
    # macro_f1 = np.mean(per_case_f1)

    # # Micro-F1 still available — pools all edges globally
    # micro_f1 = f1_score(val_edge_label, val_pred, average='micro')

    # print(f"Macro-F1 (per-case): {macro_f1:.4f}")
    # print(f"Micro-F1 (all edges): {micro_f1:.4f}")
    # print(f"AUC: {roc_auc_score(val_edge_label, val_logit):.4f}")
    # print(f"Confusion matrix:\n{confusion_matrix(val_edge_label, val_pred)}")

if __name__=='__main__':
    main()