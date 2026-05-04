import os
import pickle as pkl
import pandas as pd
from datasets import load_dataset
import spacy
from tqdm import tqdm

def main():
    
    #################################
    ## Getting terms from article_term_graph
    #################################
   
    with open('datasets/article_term_graph.pkl', 'rb') as file:
        article_term_graph = pkl.load(file)
    

    #################################
    ## Getting facts from ECthr
    #################################
    all_cases_data = load_dataset('ecthr_cases')
    df_cases = all_cases_data["train"].to_pandas()

    edge_count = 0
    

    nlp = spacy.load("en_core_web_sm")
    fact_term_edges=[]
    print(len(article_term_graph['term_index'].keys()))
    n = len(df_cases)
    fact_num = 0
    for case_num in tqdm(range(n)):
        facts = df_cases.loc[case_num, 'facts']
        for fact in facts:
            doc = nlp(fact)
            curr_nouns = set([token.text.lower() for token in doc 
                            if (token.pos_ == 'NOUN' and token.text.isalpha())])
            for noun in curr_nouns:
                ## Note we are using terms from articles only
                if(noun in article_term_graph['term_index'].keys()):
                    edge_count+=1
                    ## note multiple edges not handled
                    fact_term_edges.append([fact_num, article_term_graph['term_index'][noun]])
            fact_num+=1      
    
    fact_term_graph = {'term_index': article_term_graph['term_index'],
                        'term_reverse_index': article_term_graph['term_reverse_index'],
                        'fact_term_edges': fact_term_edges}
    print(edge_count,fact_num)
    with open('datasets/fact_term_graph_new.pkl', 'wb') as file:
        pkl.dump(fact_term_graph, file)

if __name__=='__main__':
    main()