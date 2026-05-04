import pickle as pkl
import torch
from torch_geometric.data import HeteroData
from torch_geometric.data import Data
with open('./med_exp/graph/MedGraph.pkl', 'rb') as file:
    hier_graph = pkl.load(file)

print(hier_graph)

edge_index = hier_graph['notes','links','icds'].edge_index
num_notes = hier_graph['notes'].num_nodes
num_icds = hier_graph['icds'].num_nodes
note_degrees = torch.bincount(edge_index[0], minlength=num_notes)
_ , sorted_note = torch.sort(note_degrees, descending=True)
icd_degrees = torch.bincount(edge_index[1], minlength=num_icds)
_ , sorted_icd = torch.sort(icd_degrees, descending=True)

# Select top k icd nodes
sorted_note=sorted_note[:9000]
sorted_note_new_id = {}
for i in range(len(sorted_note)):
    sorted_note_new_id[sorted_note[i].item()] = i
k = 28
sorted_icd = sorted_icd[:k]
sorted_icd_new_id = {}
for i in range(len(sorted_icd)):
    sorted_icd_new_id[sorted_icd[i].item()] = i

note_labels = {}

a_indices = edge_index[0].tolist()
b_indices = edge_index[1].tolist()


b_to_a_neighbors = {}
for i in range(len(a_indices)):
    a_idx = a_indices[i]
    b_idx = b_indices[i]

    if a_idx in sorted_note:
        if sorted_note_new_id[a_idx] not in note_labels:
            note_labels[sorted_note_new_id[a_idx]] = [0 for _ in range(k)]
        if b_idx in sorted_icd:
            note_labels[sorted_note_new_id[a_idx]][sorted_icd_new_id[b_idx]] = 1

            if b_idx not in b_to_a_neighbors:
                b_to_a_neighbors[b_idx] = []
            b_to_a_neighbors[b_idx].append(sorted_note_new_id[a_idx])

edges = []
for a_neighbors_list in b_to_a_neighbors.values():
    for i in a_neighbors_list:
            for j in a_neighbors_list:
                 if i != j:
                    edges.append((i,j))


projected_edge_index = torch.tensor(edges).t()




# new_graph = HeteroData()
# new_graph["notes"].x = hier_graph["notes"].x[sorted_note]
# new_graph["notes","connect","notes"].edge_index = projected_edge_index
# new_graph.y = torch.tensor([v for k, v in sorted(note_labels.items())], dtype=torch.long)


new_graph = Data(x=hier_graph["notes"].x[sorted_note], edge_index=projected_edge_index)
new_graph.y = torch.tensor([v for k, v in sorted(note_labels.items())], dtype=torch.long)

print(new_graph)
# new_graph = new_graph.to_homogeneous()
print(new_graph)
print(new_graph.num_features)


with open('./med_exp/medgraph_exp/MedGraph.pkl','wb') as file:
    pkl.dump(new_graph,file)