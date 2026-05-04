import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pickle as pkl
from tqdm import tqdm
import torch
torch.cuda.empty_cache()
from torch.utils.data import Dataset, DataLoader
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

def move_outputs_to_device(outputs, device):
    outputs.last_hidden_state = outputs.last_hidden_state.to(device)
    outputs.pooler_output = outputs.pooler_output.to(device)
    
    if outputs.attentions is not None:
        outputs.attentions = [attn.to(device) for attn in outputs.attentions]
    
    if outputs.cross_attentions is not None:
        outputs.cross_attentions = [cross_attn.to(device) for cross_attn in outputs.cross_attentions]
    
    return outputs

def compute_embeds(tokenizer, model, cases_list, batch_size=8):
    # Extract facts from the cases
    facts, fact_indices = [], []
    for case_id, case in enumerate(cases_list):
        case_facts_separated = [' '.join(text.split('.')[1:]).strip() for text in case]
        case_facts_indices = [(case_id, fact_id) for fact_id in range(len(case))]
        facts.extend(case_facts_separated)
        fact_indices.extend(case_facts_indices)

    # Create dataset and dataloader
    dataset = TextDataset(facts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    embeddings = []
    for batch in tqdm(dataloader):
        outputs = model(input_ids=batch['input_ids'].to(device).squeeze(), 
                        token_type_ids=batch['token_type_ids'].to(device).squeeze(), 
                        attention_mask=batch['attention_mask'].to(device).squeeze())
        outputs = move_outputs_to_device(outputs, "cpu")
        batch_embeddings = outputs.last_hidden_state.cpu()[:, 0, :]  # Using CLS token embedding
        embeddings.append(batch_embeddings.detach())
      
    embeddings = torch.cat(embeddings, dim=0)
    print(embeddings.shape)
    return fact_indices, embeddings

def main():
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
    model = BertModel.from_pretrained('nlp  aueb/legal-bert-base-uncased').to(device)
    model.eval()

    # Load ECTHR dataset and compute fact embeddings
    all_cases_data = load_dataset('ecthr_cases')
    print(all_cases_data)
    fact_embeddings_all = dict()
    fact_indices_all = dict()
    # for split in ['test', 'train', 'validation']:
    #     fact_indices, embeddings = compute_embeds(tokenizer, model, all_cases_data[split]['facts'])
    #     fact_embeddings_all[split] = embeddings
    #     fact_indices_all[split] = fact_indices
    # print(type(all_cases_data['train']['facts']),len(all_cases_data['train']['facts']))
    
    # with open('./datasets/fact_embeddings.pkl', 'wb') as file:
    #     pkl.dump(fact_embeddings_all, file)    
    # with open('./datasets/fact_indices.pkl', 'wb') as file:
    #     pkl.dump(fact_indices_all, file)
    case_list_1 = all_cases_data['train']['facts'][0:3000]
    case_list_2 = all_cases_data['train']['facts'][3000:6000]
    case_list_3 = all_cases_data['train']['facts'][6000:9000]

    fact_indices, embeddings = compute_embeds(tokenizer, model, case_list_1)
    with open('./datasets/fact_embeddings1.pkl', 'wb') as file:
        pkl.dump(embeddings, file)
    fact_indices, embeddings = compute_embeds(tokenizer, model, case_list_2)
    with open('./datasets/fact_embeddings2.pkl', 'wb') as file:
        pkl.dump(embeddings, file)
    fact_indices, embeddings = compute_embeds(tokenizer, model, case_list_3)
    with open('./datasets/fact_embeddings3.pkl', 'wb') as file:
        pkl.dump(embeddings, file)

    
    

if __name__=='__main__':
    main()
