import pickle as pkl
# import torch
import matplotlib.pyplot as plt
import numpy as np
# with open('./med_exp/graph/MedGraph.pkl', 'rb') as file:
#     hier_graph = pkl.load(file)

# print(hier_graph)

# print("Splitting data")
# import torch_geometric.transforms as T

with open('./med_exp/medgraph_exp/MedGraph.pkl', 'rb') as f:
    di = pkl.load(f)
print(di)
# print(di.node_type)
# print(di.edge_type)
sum_icds= [0 for _ in range(28)]
for i in di.y:
    labels = list(i)
    for j in range(28):
        sum_icds[j]+=labels[j].item()
for i in range(28):
    sum_icds[i] = round(sum_icds[i] /9000,3)
print(sum_icds)
# print("Splitting data")



# edge_index = hier_graph['notes', 'links', 'icds'].edge_index
# dst_nodes = edge_index[1]
# num_icds = hier_graph['icds'].num_nodes  # 8367




# 1. Calculate the degree (count of incoming note-edges) for every icd node
# # torch.bincount counts occurrences of each index up to minlength
# icd_degrees = torch.bincount(dst_nodes, minlength=num_icds)
# # 2. Sort the degrees in descending order for the frequency graph
# sorted_degrees, sorted_indices = torch.sort(icd_degrees, descending=True)
# print(len(sorted_degrees))
# print(dst_nodes)
# print(icd_degrees)
# print(sorted_degrees)
# print(sorted_indices)


# sorted_degrees = sorted_degrees[:500]
# # print(len(sorted_degrees))
# # 3. Plotting
# plt.figure(figsize=(30, 60))
# plt.plot(sorted_degrees.numpy(), color='teal', linewidth=2)
# plt.title('Frequency of Notes linked to ICDs (Sorted)')
# plt.xlabel('ICD Node Rank (Position)')
# plt.ylabel('Number of Linked Notes')
# custom_ticks = np.arange(0, 6000, 100)
# plt.yticks(custom_ticks) #
# custom_ticks = np.arange(0, 500, 50) # Ticks every 0.02 units
# plt.xticks(custom_ticks) #
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# # plt.show()

# plt.savefig('./med_exp/medgraph_exp/icd_edge_frequency.png', dpi=300, bbox_inches='tight')

# print("Graph saved as 'icd_edge_frequency.png'")