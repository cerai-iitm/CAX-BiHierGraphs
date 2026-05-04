import os
import pickle as pkl
from tqdm import tqdm
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def main():
    ##############################################
    # Load the article-term, case-fact graph  and fact-term graphs
    ##############################################
    article_term_file = './datasets/article_term_graph.pkl'
    fact_term_file = './datasets/fact_term_graph_new.pkl'
    case_fact_file = './datasets/case_fact_graph.pkl'
    case_article_file = './datasets/case_article_graph.pkl'
    with open(article_term_file, 'rb') as file:
        article_term_graph = pkl.load(file)
    with open(fact_term_file, 'rb') as file:
        fact_term_graph = pkl.load(file)
    with open(case_fact_file, 'rb') as file:
        case_fact_graph = pkl.load(file)
    with open(case_article_file, 'rb') as file:
        case_article_graph = pkl.load(file)
    ####################################################
    # Load the embeddings for terms, articles and facts
    ####################################################
    article_embed_file = './datasets/article_embeddings_all.pkl' 
    term_embed_file = './datasets/term_embeddings_all.pkl'
    fact_embed_file1 = './datasets/fact_embeddings1.pkl'
    fact_embed_file2 = './datasets/fact_embeddings2.pkl'
    fact_embed_file3 = './datasets/fact_embeddings3.pkl'

    with open(article_embed_file, 'rb') as file:
        article_embeds = pkl.load(file)
    with open(term_embed_file, 'rb') as file:
        term_embeds = pkl.load(file)
    with open(fact_embed_file1, 'rb') as file:
        fact_embeddings1 = pkl.load(file)
    with open(fact_embed_file2, 'rb') as file:
        fact_embeddings2 = pkl.load(file)
    with open(fact_embed_file3, 'rb') as file:
        fact_embeddings3 = pkl.load(file)
    # #################################
    # we have used only facts/cases from train part of dataset
    #################################
    # Construct hierarchical graph
    #################################
    data = HeteroData()

    #NODE declarations
    data['articles'].x = article_embeds['embeddings'].cpu()
    data['terms'].x = term_embeds['embeddings'].cpu()
    
   
    fact_embeds = torch.cat((fact_embeddings1,fact_embeddings2,fact_embeddings3), dim=0)
    data['facts'].x = fact_embeds

    num_cases = 9000
    case_embeds = torch.zeros([num_cases,768])
    data['cases'].x = case_embeds


    #EDGE declarations
    data['articles','has', 'terms'].edge_index = torch.tensor(article_term_graph['article_term_edges']).t().contiguous() # [2, num_edges_cites] 
    data['facts','has', 'terms'].edge_index = torch.tensor(fact_term_graph['fact_term_edges']).t().contiguous()
    data['cases','violate','articles'].edge_index = torch.tensor(case_article_graph).t().contiguous()
    data['facts','part_of', 'cases'].edge_index = torch.tensor(case_fact_graph['case_fact_edges']).t().contiguous()
    print(case_embeds.shape)
    for node_type in ['articles', 'terms', 'facts', 'cases']:
        data[node_type, 'self_{}'.format(node_type), node_type].edge_index = \
                                                    torch.stack([torch.arange(data[node_type].x.shape[0]),
                                                                 torch.arange(data[node_type].x.shape[0])], dim=0)
    # automatically copies edges in reverse to make it undirected graph
    #data = T.AddSelfLoops()(data)
    
    data = T.ToUndirected()(data) 
    print(data,data.has_self_loops())
    
    with open('./datasets/Hetero_Data_With_Self_Loops.pkl', 'wb') as file:
        pkl.dump(data, file)



if __name__=='__main__':
    main()
