import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import pickle as pkl
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd

print('Num devices= ', torch.cuda.device_count())
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
print(device)

def compute_embeds(tokenizer, model, terms, batch_size=1024):
    print("start")
    

    tokenized_terms = tokenizer(terms, return_tensors='pt',padding=True
                                        ).to(device)
    input_ids = tokenized_terms['input_ids']
    attention_mask = tokenized_terms['attention_mask']

    
    # for i in range(440):
    #     decoded_text = tokenizer.decode(input_ids[i])
    #     print(f"Decoded Text: {decoded_text}")
    #     tokenized_text = tokenizer.tokenize(decoded_text)
    #     print(f"tokenized Text: {tokenized_text}")
    # print(input_ids.shape)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]
    print(embeddings.shape,"end")
    return embeddings

def main():
    # Load pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Charangan/MedBERT")
    model = AutoModel.from_pretrained("Charangan/MedBERT").to(device)


    # tokenizer = BertTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
    # model = BertModel.from_pretrained('nlpaueb/legal-bert-base-uncased').to(device)
    model.eval()

    
    df = pd.read_csv("../../data/icd10descriptions.csv", encoding='latin-1')

    icd_desc = []

    for i,j in df.iterrows():
        icd_desc.append(j['long_title'] + '\n' + j['Detailed Description'])

    embeddings = compute_embeds(tokenizer, model,icd_desc,batch_size=440)
    with open('./icd_embeddings.pkl', 'wb') as file:
        pkl.dump(embeddings, file)

if __name__=='__main__':
    main()
