import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pickle as pkl
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def main():
    article_file_path = '/home/gokul/LegalGraph/code/Article-Data/ECHR_Articles_new.csv'

    df = pd.read_csv(article_file_path, delimiter='\t')
    print(df.head())

    model_path = "nlpaueb/bert-base-uncased-echr"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model = model.to(device)
    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")
    term_list = set()

    for i in range(len(df)):
        doc = nlp(df.loc[i, 'Description'])
        curr_nouns = set([token.text.lower() for token in doc 
                          if (token.pos_ == 'NOUN' and token.text.isalpha())])
        term_list.update(curr_nouns)
        #text_tokens = tokenizer.tokenize(df.loc[i, 'Description'])
        #print(text_tokens)
    print(term_list)
    print(len(term_list))
    term_index = {term: ind for ind, term in enumerate(list(term_list))}
    term_reverse_index = {ind: term for ind, term in enumerate(list(term_list))}

    article_term_edges = []
    for i in range(len(df)):
        doc = nlp(df.loc[i, 'Description'])
        curr_nouns = set([token.text.lower() for token in doc 
                          if (token.pos_ == 'NOUN' and token.text.isalpha())])
        for noun in curr_nouns:
            article_term_edges.append([i, term_index[noun]])
    
    article_term_graph = {'term_index': term_index,
                          'term_reverse_index': term_reverse_index,
                          'article_term_edges': article_term_edges}
    with open('datasets/article_term_graph.pkl', 'wb') as file:
        pkl.dump(article_term_graph, file)

if __name__=='__main__':
    main()
