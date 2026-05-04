# this code reads the existing medgraph and removes the duplicate edges
import pickle as pkl
import torch


graph  = pkl.load(open("med_exp/graph/MedGraph.pkl","rb"))
print(graph)
for edge_type, edge_index in graph.edge_index_dict.items():
    edges_rows = edge_index.t()
    unique_edges_rows = torch.unique(edges_rows, dim=0, sorted=False)
    unique_edge_index = unique_edges_rows.t()
    graph[edge_type].edge_index = unique_edge_index
print(graph)
with open("med_exp/graph/MedGraph_NoDupEdges.pkl","wb") as file:
    pkl.dump(graph,file)