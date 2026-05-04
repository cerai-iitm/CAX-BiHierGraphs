import pickle as pkl


with open('./med_exp/graph/MedGraph.pkl', 'rb') as file:
    hier_graph = pkl.load(file)


print(hier_graph)
retain_nodes = ['icds','notes']
for i in hier_graph.node_types:
    if i not in retain_nodes:
        del hier_graph[i]
for i in hier_graph.edge_types:
    if i[0] not in retain_nodes or  i[2] not in retain_nodes:
        del hier_graph[i]

print(hier_graph)

with open('./med_exp/heteromed/HeteroMedGraph.pkl','wb') as file:
    pkl.dump(hier_graph,file)