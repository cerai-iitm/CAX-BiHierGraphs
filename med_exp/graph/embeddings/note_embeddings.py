import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import pickle as pkl
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
print('Num devices= ', torch.cuda.device_count())
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
print(device)

def compute_embeds(tokenizer, model, terms, batch_size=1024):
    
    

    tokenized_terms = tokenizer(terms, return_tensors='pt',padding=True
                                        ).to(device)
    input_ids = tokenized_terms['input_ids']
    attention_mask = tokenized_terms['attention_mask']

    
    # for i in range(440):
    #     decoded_text = tokenizer.decode(input_ids[i])
    #     print(f"Decoded Text: {decoded_text}")
    #     tokenized_text = tokenizer.tokenize(decoded_text)
    #     print(f"tokenized Text: {tokenized_text}")


    # cutting tokens to fit context window
    if(input_ids.shape[1] > 512):
        input_ids = input_ids[:,:512]
        attention_mask = attention_mask[:,:512]

        

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]
    
    return embeddings

def main():
    # Load pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Charangan/MedBERT")
    model = AutoModel.from_pretrained("Charangan/MedBERT").to(device)


    # tokenizer = BertTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
    # model = BertModel.from_pretrained('nlpaueb/legal-bert-base-uncased').to(device)
    model.eval()

    df = pd.read_csv("../../data/notes.csv",usecols=["Chief Complaint","Past Medical History"])

    note_desc= []
    for i,j in df.iterrows():
        if isinstance(j['Chief Complaint'], str) and isinstance(j['Past Medical History'], str):
            note_desc.append('Chief Complaint : '+j['Chief Complaint'] + '\nPast Medical History : ' + j['Past Medical History'])
        elif isinstance(j['Chief Complaint'], str):
            note_desc.append('Chief Complaint : '+j['Chief Complaint'] )
        elif isinstance(j['Past Medical History'], str):
            note_desc.append('Past Medical History : ' + j['Past Medical History'])
        else:
            note_desc.append("")
   
    
    i=0

    for i in tqdm(range(0,len(note_desc),1000)):
        note_desc_i = note_desc[i:min(i+1000,len(note_desc))]
        embeddings = compute_embeds(tokenizer, model,note_desc_i,batch_size=440)
        if i >0 :
            with open(f'./note_embeddings.pkl', 'rb') as file:
                prev_embed = pkl.load(file)
            embeddings = torch.cat((prev_embed,embeddings), dim=0)
        with open(f'./note_embeddings.pkl', 'wb') as file:
                pkl.dump(embeddings, file)
        # print(embeddings.shape)
        del embeddings


    
    
    # embeddings = compute_embeds(tokenizer, model,note_des1,batch_size=440)
    # with open('./note_embeddings1.pkl', 'wb') as file:
    #     pkl.dump(embeddings, file)
    # del embeddings
    

if __name__=='__main__':
    main()
