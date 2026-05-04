import pickle as pkl
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
device = torch.device('cuda') if (torch.cuda.is_available()) else torch.device('cpu') 
# device = torch.device('cpu') 
print(device)
from Explainer_utils import hetero_src_tgt_khop_in_subgraph, k_shortest_paths_with_max_length
# from graph_visualizer import visualize
from tqdm import tqdm
import pickle as pkl
from Graph_utils import *
from get_human_readable_graph_explanations import get_LLM_explanations_all, get_LLM_base_explanation

explanation_path = 'results/explanations/hetero_pg_explainer/pos_explanations.pkl'

def getFeat(graph,mapping):
    temp = {}
    for node_type in graph.node_types:
         for i, node in enumerate(graph[node_type].x):
            if node_type == 'facts' :
                temp[(node_type,i)] = getFact(mapping[node_type][i])
            elif node_type == 'terms':
                temp[(node_type,i)] = getTerm(mapping[node_type][i])
            elif node_type == 'articles':
                temp[(node_type,i)] = getArticle(mapping[node_type][i])
            elif node_type == 'cases':
                temp[(node_type,i)] = f"Case_{i}\n"
                facts = df_cases.loc[mapping[node_type][i], 'facts']
                for fact in facts:
                    temp[(node_type,i)] += fact + '\n'
    return temp
# def getSelectedFacts(comp_g, comp_g_src_nid, mapping):
#     req_indices = (comp_g[("facts", "part_of", "cases")].edge_index[1] == comp_g_src_nid).nonzero().reshape(-1)
#     facts_comp_g = torch.index_select(comp_g[("facts", "part_of", "cases")].edge_index,1,req_indices)[0]
#     facts_graph= []
#     for i in range(facts_comp_g.shape[0]):
#         temp = mapping["facts"][facts_comp_g[i].item()]
#         facts_graph.append(fact_case_offset[temp][0])

#     return facts_graph
s_ntype = 'cases'
t_ntype = 'articles' 
num_hops = 3
counter = -1
    
# store selected facts to calculate a retrieval metrics
# selected_facts = {}

graph = pkl.load(open('results/explanations/hetero_gnn_explainer/graph.pkl', 'rb'))

graph.to(device)


with open(explanation_path, 'rb') as file:
    explanations = pkl.load(file)

links_to_explain = [3, 7, 10, 13, 14, 16, 18, 21, 404, 407, 413, 414, 416, 1823, 2549, 2550]


for nodes,mask in tqdm(explanations.items()):
        counter += 1
        print(counter,end=" ")

        # debug later
        if counter ==3:
            continue
        
        if counter not in links_to_explain:
            continue

        s_nid = nodes[0]
        t_nid = nodes[1]
        
        eweight_dict = {etype: mask[etype].detach().sigmoid().to(device) for etype in mask}      
        print("\nfinding khop",counter)
        (comp_g_src_nid, 
                comp_g_tgt_nid, 
                comp_g_after_khop, mapping
                ) = hetero_src_tgt_khop_in_subgraph(s_ntype, 
                                                    s_nid, 
                                                    t_ntype, 
                                                    t_nid, 
                                                    graph, 
                                                    num_hops,
                                                    device)
    #    Beam search
        print("finding beam search",counter)
        (comp_g_src_nid, 
                comp_g_tgt_nid, 
                comp_g, masked_mapping
                ) = hetero_src_tgt_khop_in_subgraph(s_ntype, 
                                                    comp_g_src_nid, 
                                                    t_ntype, 
                                                    comp_g_tgt_nid, 
                                                    comp_g_after_khop, 
                                                    num_hops,
                                                    device,
                                                    eweight_dict,
                                                    2)
            
        # get the subgraph node to original graph node mapping
        for ntype in mapping:
            rev_map = {sub: orig for orig, sub in mapping[ntype].items()}
            mapping[ntype] = rev_map
        for ntype in masked_mapping:
            rev_map = {sub: orig for orig, sub in masked_mapping[ntype].items()}
            masked_mapping[ntype] = rev_map

        # We need one map from subgraph to original graph
        original_mapping={}
        for ntype in masked_mapping:
            original_mapping[ntype] = {sub: mapping[ntype][orig] for sub,orig in masked_mapping[ntype].items()}
        
    
        # Get features of nodes
        # feat_nodes = getFeat(comp_g,original_mapping)
        
        

        # get the selected facts
        # if s_nid in selected_facts:
        #     selected_facts[s_nid].update(set(getSelectedFacts(comp_g, comp_g_src_nid, mapping)))
        # else:
        #     selected_facts[s_nid] = set(getSelectedFacts(comp_g, comp_g_src_nid, mapping))
             
        
        # Get weights of edges after beam search
        # for edge_type in comp_g.edge_types:
        #     comp_g[edge_type]["mask"]=[]
        #     src_type, rel_type, dst_type = edge_type
        #     edge_index = comp_g[edge_type].edge_index
        #     edge_index_after_khop = comp_g_after_khop[edge_type].edge_index
            
        #     for i in range(edge_index.size(1)):
        #         src, dst = edge_index[:, i].tolist()
        #         src = masked_mapping[src_type][src]
        #         dst = masked_mapping[dst_type][dst]
                
        #         mask_index = (edge_index_after_khop[0] == src) & (edge_index_after_khop[1] == dst)
        #         edge_idx = mask_index.nonzero(as_tuple=False).squeeze().item()
        #         comp_g[edge_type]["mask"].append(eweight_dict[edge_type][edge_idx].item())
        #         # comp_g[edge_type]["mask"].append(round(eweight_dict[edge_type][edge_idx].item(),2))
                
        print("finding k_shortest_paths_with_max_length",counter)
        # print('cases',
        #         comp_g_src_nid,
        #         'articles',
        #         comp_g_tgt_nid)
        explaination_paths = k_shortest_paths_with_max_length(
                comp_g,
                'cases',
                comp_g_src_nid,
                'articles',
                comp_g_tgt_nid
        )

        # for pidx, path in enumerate(explaination_paths):
        #         for eidx, edge in enumerate(path):
        #             print(edge,original_mapping[edge_type[0]])
        #             src_nid , tgt_nid, edge_type = edge
        #             explaination_paths[pidx][eidx] = (original_mapping[edge_type[0]][src_nid], original_mapping[edge_type[2]][tgt_nid], edge_type)
        # print(original_mapping)
        key_errors = set()
        for pidx, path in enumerate(explaination_paths):
            for eidx, edge in enumerate(path):
                src_nid, tgt_nid, edge_type = edge
                # print(edge)
                if edge_type == '_':
                    key_errors.add(pidx)
                    continue
                try:
                    explaination_paths[pidx][eidx] = (original_mapping[edge_type[0]][src_nid], original_mapping[edge_type[2]][tgt_nid], edge_type)
                except KeyError:
                    key_errors.add(pidx)
        explaination_paths = [path for i, path in enumerate(explaination_paths) if i not in key_errors]
        # print('Key errors: ', key_errors)
        
        print("finding path",counter)
        all_human_readable_explanations = get_LLM_explanations_all(explaination_paths)
        # for human_readable_explanation in all_human_readable_explanations:
        #     with open(f'results/explanations/hetero_pg_explainer/pgexp{counter}.txt', 'a') as f:
        #         f.write(human_readable_explanation)
        # for idx, human_readable_explanation in enumerate(all_human_readable_explanations):
        #         if idx not in key_errors:
        #             print("writing to file for counter",counter)
        #             with open(f'results/explanations/hetero_pg_explainer/pgexp{counter}.txt', 'a') as f:
        #                 f.write(human_readable_explanation)     
        # ensure that html file in argument given doesnt already exist
        # If include_edge_weight is True make sure to give edge weights as an edge attribute of comp_g called 'mask'
        # i.e, comp_g[edge_type]["mask"][i] should have weight of ith edge of type edge_type
        # visualize(comp_g,
        #           s_ntype, 
        #           comp_g_src_nid, 
        #           t_ntype, 
        #           comp_g_tgt_nid,
        #           feat_nodes,
        #           f'results/explanations/hetero_pg_explainer/pgexp{counter}.html',
        #           )
        # break
        

# pkl.dump(selected_facts, open('results/explanations/hetero_pg_explainer/selected_facts_pgexp.pkl', 'wb'))
          
        
        
