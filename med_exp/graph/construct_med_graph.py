import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import pickle as pkl
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def main():
    ##############################################
    # Load the article-term, case-fact graph  and fact-term graphs
    ##############################################
    note_term_file = './construction/note_term_graph.pkl'
    icd_term_file = './construction/icd_term_graph.pkl'
    note_icd_file = './construction/note_icd_graph.pkl'

    
    with open(note_term_file, 'rb') as file:
        note_term_graph = pkl.load(file)
    with open(icd_term_file, 'rb') as file:
        icd_term_graph = pkl.load(file)
    with open(note_icd_file, 'rb') as file:
        note_icd_graph = pkl.load(file)


    ####################################################
    # Load the embeddings for terms, articles and facts
    ####################################################
    note_embed_file = './embeddings/note_embeddings.pkl' 
    icd_embed_file = './embeddings/icd_embeddings.pkl' 
    term_embed_file = './embeddings/terms_embeddings.pkl' 

    with open(note_embed_file, 'rb') as file:
        note_embeddings = pkl.load(file)
    with open(icd_embed_file, 'rb') as file:
        icd_embeddings = pkl.load(file)
    with open(term_embed_file, 'rb') as file:
        terms_embeddings = pkl.load(file)

    
    #################################
    # Construct hierarchical graph
    #################################
    data = HeteroData()

    #NODE declarations
    data['icds'].x = icd_embeddings.cpu()
    data['terms'].x = terms_embeddings.cpu()
    data['notes'].x = note_embeddings.cpu()


    # #EDGE declarations
    data['notes','has', 'terms'].edge_index = torch.tensor(note_term_graph["note_term_edges"]).t().contiguous() # [2, num_edges_cites] 
    data['icds','has', 'terms'].edge_index = torch.tensor(icd_term_graph["icd_term_edges"]).t().contiguous()
    data['notes','links','icds'].edge_index = torch.tensor(note_icd_graph["note_icd_edges"]).t().contiguous()
    # print(case_embeds.shape)
    for node_type in ['notes', 'icds', 'terms']:
        data[node_type, 'self_{}'.format(node_type), node_type].edge_index = \
                                                    torch.stack([torch.arange(data[node_type].x.shape[0]),
                                                                 torch.arange(data[node_type].x.shape[0])], dim=0)
    # # automatically copies edges in reverse to make it undirected graph
    # #data = T.AddSelfLoops()(data)
    
    data = T.ToUndirected()(data) 
    # print(data,data.has_self_loops())
    print(data)

    
    with open('./MedGraph.pkl', 'wb') as file:
        pkl.dump(data, file)

if __name__=='__main__':
    main()
