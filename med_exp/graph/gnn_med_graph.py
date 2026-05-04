import os
import json
import pickle as pkl
import numpy as np
from collections import Counter
from tqdm import tqdm
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, GraphConv, SAGEConv, to_hetero
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric import seed_everything

from sklearn.metrics import confusion_matrix, roc_auc_score
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

class HierGNN_2_Layer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # self.conv1 = GATConv(input_dim, hidden_dim)
        # self.conv2 = GATConv(input_dim, hidden_dim)
        self.conv1 = GATConv(input_dim, hidden_dim, add_self_loops=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, add_self_loops=False)
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
    def __init__(self, graph, input_dim, hidden_dim,layers=3):
        super().__init__()
        # self.case_embed = torch.nn.Parameter(case_attr)
        # Set the embedding weights to pre-computed values
        #self.case_embed.weight.data.copy_(case_attr)

        # Instantiate homogeneous GNN:
        self.gnn = None
        if layers==3:
            self.gnn = HierGNN(input_dim, hidden_dim)
        else:
            self.gnn = HierGNN_2_Layer(input_dim, hidden_dim)



        # Convert GNN model into a heterogeneous variant:
        #print(graph.metadata())
        self.gnn = to_hetero(self.gnn, metadata=graph.metadata())
        #print('Hetero GNN:\n', self.gnn)
        self.classifier = Classifier()

    def forward(self, graph, eweights=None):
        x_dict = {
          "icds": graph["icds"].x,
          "terms": graph["terms"].x,
          "notes": graph["notes"].x,
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
            x_dict["notes"],
            x_dict["icds"],
            graph["notes", "links", "icds"].edge_label_index,
        )
        return pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help = '''Path to the saved hierarchical graph''', 
                        default = './med_exp/graph/MedGraph.pkl', type = str)
    # parser.add_argument('--case_embeds_path', help = '''Path to the saved case embeddings''', 
    #                     default = '/home/gokul/LegalGraph/code/dumps/legalbert_embeds_train.pkl', type = str)      
    parser.add_argument('--results_path', help = '''Path to save the results''',
                        default = './results', type = str)
    parser.add_argument('--model_save_path', help = '''Path to save the trained model''',
                        default = './results/hetero_gnn_model_neg_10.pt', type = str)
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
    parser.add_argument('--layers', help = 'Number of layers for the model', 
                        default = 3, type = int) 
    parser.add_argument('--device', help = 'Device to be used for model training', 
                        default = 'cuda', choices = ['cuda', 'cpu'], type = str) 
    args = parser.parse_args()
    torch_device = torch.device('cuda') if (torch.cuda.is_available() and args.device == 'cuda') \
        else torch.device('cpu') 
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
    
    # with open(args.case_embeds_path, 'rb') as file:
    #     case_embeds_legalbert = pkl.load(file)
    #print(type(case_embeds_legalbert), len(case_embeds_legalbert))
    # case_embeds_legalbert = torch.cat(case_embeds_legalbert, dim=0)
    
    transform = T.RandomLinkSplit(
                    num_val=0.2,
                    num_test=0.2,  
                    disjoint_train_ratio=0.3,
                    neg_sampling_ratio=args.neg_sampling_ratio,
                    # add_negative_train_samples=False, 
                    edge_types=('notes','links','icds'),
                    rev_edge_types=("icds", "rev_links", "notes"),)
    train_data, val_data, test_data = transform(hier_graph)
    print(train_data)

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
    model = Model(hier_graph, hier_graph['notes'].x.shape[1], args.hidden_dim, args.layers)
    model.to(torch_device)
    model.train()
    #print(model)
    train_data = train_data.to(torch_device)
    train_edge_label_index = train_data["notes", "links", "icds"].edge_label_index.to(torch_device)
    train_edge_label = train_data["notes", "links", "icds"].edge_label.to(torch_device)

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
    if args.layers==3:
        with open(os.path.join(args.results_path, 'epochwise_losses.pkl'), 'wb') as file:
           pkl.dump(epochwise_losses, file)
        torch.save(model.state_dict(), args.model_save_path)
    else:
        with open(os.path.join(args.results_path, 'epochwise_losses_2_layer.pkl'), 'wb') as file:
           pkl.dump(epochwise_losses, file)
        torch.save(model.state_dict(), args.model_save_path[:-3]+"_2_layer.pt")
        
    
    
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
    val_edge_label_index = val_data["notes", "links", "icds"].edge_label_index
    val_edge_label = val_data["notes", "links", "icds"].edge_label

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
    val_pred = (val_pred > 0) 
    val_edge_label = val_edge_label.detach().cpu().numpy()
    print(val_pred.shape, val_edge_label.shape)
    print(val_pred)
    print(val_edge_label)
    print(f"\nValidation AUC: {roc_auc_score(val_edge_label, val_logit):.4f}")
    print("Validation confusion matrix =\n", confusion_matrix(val_edge_label, val_pred))
    print('============================================================')


    if args.layers==3:
        with open(os.path.join(args.results_path, 'val_predictions.pkl'), 'wb') as file:
            pkl.dump({'ground_truth': val_edge_label,
                    'pred': val_logit}, file)
        with open(os.path.join(args.results_path, 'train_predictions.pkl'), 'wb') as file:
            pkl.dump({'ground_truth': train_edge_label,
                    'pred': train_logit}, file) 
    else:
        with open(os.path.join(args.results_path, 'val_predictions_2_layer.pkl'), 'wb') as file:
            pkl.dump({'ground_truth': val_edge_label,
                    'pred': val_logit}, file)
        with open(os.path.join(args.results_path, 'train_predictions_2_layer.pkl'), 'wb') as file:
            pkl.dump({'ground_truth': train_edge_label,
                    'pred': train_logit}, file) 
     

if __name__=='__main__':
    main()