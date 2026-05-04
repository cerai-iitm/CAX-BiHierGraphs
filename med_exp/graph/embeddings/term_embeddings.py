import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import pickle as pkl
import torch
from transformers import AutoTokenizer, AutoModel


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

    #################################
    ## Getting terms from article_term_graph
    #################################
    with open('../construction/term_index.pkl', 'rb') as file:
        term_index = pkl.load(file)
    terms = list(term_index['term_for_index'].keys())
    embeddings = compute_embeds(tokenizer, model,terms,batch_size=440)
    with open('./terms_embeddings.pkl', 'wb') as file:
        pkl.dump(embeddings, file)
    
    # terms1 = terms

    # print(len(terms),len(terms1),len(terms2),len(terms3))

    # # embeddings = compute_embeds(tokenizer, model,terms1,batch_size=440)
    # # with open('./terms_embeddings1.pkl', 'wb') as file:
    # #     pkl.dump(embeddings, file)
    # # del embeddings
    # # embeddings = compute_embeds(tokenizer, model,terms2,batch_size=440)
    # # with open('./terms_embeddings2.pkl', 'wb') as file:
    # #     pkl.dump(embeddings, file)
    # # del embeddings
    # # embeddings = compute_embeds(tokenizer, model,terms3,batch_size=440)
    # # with open('./terms_embeddings3.pkl', 'wb') as file:
    # #     pkl.dump(embeddings, file)

if __name__=='__main__':
    main()
