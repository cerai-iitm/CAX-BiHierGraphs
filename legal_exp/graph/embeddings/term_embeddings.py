import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pickle as pkl
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertModel

print('Num devices= ', torch.cuda.device_count())
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
print(device)

def compute_embeds(tokenizer, model, term_index, batch_size=1024):

    term_indices = torch.Tensor(list(term_index.values()))
    print(term_indices)
    terms = list(term_index.keys())
    

    tokenized_terms = tokenizer(terms, return_tensors='pt',padding=True
                                        ).to(device)
    input_ids = tokenized_terms['input_ids']
    attention_mask = tokenized_terms['attention_mask']

    
    # for i in range(440):
    #     decoded_text = tokenizer.decode(input_ids[i])
    #     print(f"Decoded Text: {decoded_text}")
    #     tokenized_text = tokenizer.tokenize(decoded_text)
    #     print(f"tokenized Text: {tokenized_text}")
    print(input_ids.shape)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]
    print(embeddings.shape)
    
    return term_indices, embeddings

def main():
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
    model = BertModel.from_pretrained('nlpaueb/legal-bert-base-uncased').to(device)
    model.eval()

    #################################
    ## Getting terms from article_term_graph
    #################################
    with open('datasets/article_term_graph.pkl', 'rb') as file:
        article_term_graph = pkl.load(file)
    term_embeddings = []
    term_indices = []
    term_indices,term_embeddings = compute_embeds(tokenizer, model,article_term_graph['term_index'],batch_size=440)
    print(term_indices.shape,term_embeddings.shape)

    term_embeddings_all=dict()
    term_embeddings_all["indices"] =term_indices
    term_embeddings_all["embeddings"] =term_embeddings
    
    print(term_embeddings_all)

    # with open('./datasets/term_embeddings_all.pkl', 'wb') as file:
    #     pkl.dump(term_embeddings_all, file)    
    

if __name__=='__main__':
    main()
