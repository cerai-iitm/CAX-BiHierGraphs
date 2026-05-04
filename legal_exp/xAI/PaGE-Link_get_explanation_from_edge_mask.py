### Apply the edge masks to the original graph to get the subgraph explanation

import torch
from torch_geometric import seed_everything
from Explainer_utils import hetero_src_tgt_khop_in_subgraph, k_shortest_paths_with_max_length
import pickle
import pandas as pd
from datasets import load_dataset
from graph_visualizer import visualize

seed_everything(4321)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

edge_masks = pickle.load(open('results/explanations/pagelink/pagelink_gnn_hier_graph_3GATConv_pred_edge_to_comp_g_edge_mask.pkl', 'rb'))
hier_graph = pickle.load(open('datasets/Hetero_Data_With_Self_Loops.pkl', 'rb'))
graph = pickle.load(open('results/explanations/pagelink/pagelink_graph.pkl', 'rb'))

article_file_path = '/home/gokul/Hier-Legal-Graph/datasets/ECHR_Articles_new.csv'
article_term_graph_path = 'datasets/article_term_graph.pkl'
case_fact_graph_file = './datasets/case_fact_graph.pkl'

# Getting all case data
all_cases_data = load_dataset('ecthr_cases', 'violation-prediction')
df_cases = all_cases_data["train"].to_pandas()

# Getting case_fact edges
with open(case_fact_graph_file, 'rb') as file:
        case_fact_graph = pickle.load(file)
    
# Getting terms from article_term_graph
with open(article_term_graph_path, 'rb') as file:
    article_term_graph = pickle.load(file)

# Getting articles
df_articles = pd.read_csv(article_file_path, delimiter='\t')

# Retriving node text from node Id
fact_case_offset = {}
offset = 0
prev_case =-1
for i in range(len(case_fact_graph['case_fact_edges'])):
    if case_fact_graph['case_fact_edges'][i][1] != prev_case:
         offset=0
    fact_case_offset[i] = [offset,case_fact_graph['case_fact_edges'][i][1]]
    prev_case = case_fact_graph['case_fact_edges'][i][1]
    offset+=1
del case_fact_graph
def getFact(node_id):
    offset = fact_case_offset[node_id][0]
    case_num = fact_case_offset[node_id][1]
    return df_cases.loc[case_num, 'facts'][offset]
def getTerm(node_id):
    return article_term_graph['term_reverse_index'][node_id]
def getArticle(node_id):
    if type(df_articles.loc[node_id, 'Title']) is float:
        df_articles.loc[node_id, 'Title'] = ''
    return "Title : "+ df_articles.loc[node_id, 'Title'] + "\nDescription : "+df_articles.loc[node_id, 'Description']

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
def getSelectedFacts(comp_g, comp_g_src_nid, mapping):
    req_indices = (comp_g[("facts", "part_of", "cases")].edge_index[1] == comp_g_src_nid).nonzero().reshape(-1)
    facts_comp_g = torch.index_select(comp_g[("facts", "part_of", "cases")].edge_index,1,req_indices)[0]
    facts_graph= []
    for i in range(facts_comp_g.shape[0]):
        temp = mapping["facts"][facts_comp_g[i].item()]
        facts_graph.append(fact_case_offset[temp][0])
    
    return facts_graph

if __name__ == "__main__":
    
    links = [3, 7, 10, 13, 14, 16, 18, 21, 404, 407, 413, 414, 416, 1823, 2549, 2550]
    counter = -1
    
    # store selected facts to calculate a retrieval metrics
    selected_facts = {}

    for nodes, paths in edge_masks.items():
        
        counter += 1
        print(counter)
        
        src_ntype, src_nid = nodes[0][0], nodes[0][1].item()
        tgt_ntype, tgt_nid = nodes[1][0], nodes[1][1].item()
        
        # get the k-hop subgraph
        (comp_g_src_nid,
        comp_g_tgt_nid,
        comp_g_k_hop,
        mapping) = hetero_src_tgt_khop_in_subgraph(src_ntype,
                                                src_nid,
                                                tgt_ntype,
                                                tgt_nid,
                                                graph,
                                                3,
                                                device=device)
        
        # get the subgraph node to original graph node mapping
        for ntype in mapping:
            rev_map = {sub: orig for orig, sub in mapping[ntype].items()}
            mapping[ntype] = rev_map
            
        # get the selected facts
        if src_nid in selected_facts:
            selected_facts[src_nid].update(set(getSelectedFacts(comp_g_k_hop, comp_g_src_nid, mapping)))
        else:
            selected_facts[src_nid] = set(getSelectedFacts(comp_g_k_hop, comp_g_src_nid, mapping))
        
        print(paths)
        # visualize the explanation subgraph
        feat_nodes = getFeat(comp_g_k_hop, mapping)
        # visualize(comp_g,
        #         'cases', comp_g_src_nid, 
        #         'articles', comp_g_tgt_nid, 
        #         feat_nodes, f'results/explanations/hetero_gnn_explainer/gnnexp{counter}.html'
        #         )
        break
            
    # pickle.dump(selected_facts, open('results/explanations/hetero_gnn_explainer/selected_facts_gnnexp.pkl', 'wb'))