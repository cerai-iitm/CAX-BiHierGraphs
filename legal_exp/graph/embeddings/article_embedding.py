import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pickle as pkl
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
torch.cuda.empty_cache()
from torch.utils.data import Dataset, DataLoader
# from torch_geometric.data import HeteroData
from datasets import load_dataset
from transformers import BertTokenizer, BertModel

print('Num devices= ', torch.cuda.device_count())
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
print(device)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        #tokenized_text = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, truncation=True)
        tokenized_text = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, 
                                        padding='max_length', truncation=True)
        return tokenized_text

def compute_embeds(tokenizer, model, df, batch_size=58):
    articles = []
    for i in range(len(df)):
        Title = df.loc[i, 'Title']
        Description = df.loc[i, 'Description']
        articles.append(Title + " " + Description )
    # print(articles[0:3])
        
    # Create dataset and dataloader
    dataset = TextDataset(articles, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    embeddings = []
    for batch in tqdm(dataloader):
        print(batch['input_ids'].shape)
        outputs = model(input_ids=batch['input_ids'].to(device).squeeze(), 
                        token_type_ids=batch['token_type_ids'].to(device).squeeze(), 
                        attention_mask=batch['attention_mask'].to(device).squeeze())
        batch_embeddings = outputs.last_hidden_state[:, 0, :]  # Using CLS token embedding
        embeddings.append(batch_embeddings)
    embeddings = torch.cat(embeddings, dim=0)
    print(embeddings.shape)
    return embeddings

def main():
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
    model = BertModel.from_pretrained('nlpaueb/legal-bert-base-uncased').to(device)
    model.eval()


    article_file_path = '/home/gokul/Hier-Legal-Graph/datasets/ECHR_Articles_new.csv'

    df = pd.read_csv(article_file_path, delimiter='\t')
    df["Title"] = df["Title"].fillna(value="")
    

    article_indices = torch.arange(0,len(df))
    print(article_indices.shape)
    
    embeddings = compute_embeds(tokenizer,model,df)

    article_embeddings_all=dict()
    article_embeddings_all["indices"] =article_indices
    article_embeddings_all["embeddings"] =embeddings

    # print(article_embeddings_all)
    with open('./datasets/article_embeddings_all.pkl', 'wb') as file:
        pkl.dump(article_embeddings_all, file)    
      

if __name__=='__main__':
    main()
