import pandas as pd
import pickle as pkl
from collections import defaultdict


file = "icd_id_to_index.pkl"
with open(file,'rb') as f:
    icd_id_to_index = pkl.load(f)

# Note: remove the argument nrows
df = pd.read_csv("../../data/notes.csv",usecols=["note_id","unique_icd_codes"])


note_id_to_index={}
note_icd_graph={}
note_icd_graph["note_icd_edges"] = []

for i,j in df.iterrows():
    note_id_to_index[j['note_id']] = i
    icd_codes = j["unique_icd_codes"].split(",")
    for icd_code in icd_codes:
        note_icd_graph["note_icd_edges"].append((i,icd_id_to_index[icd_code.strip()]))


file = "note_icd_graph.pkl"
with open(file,'wb') as f:
    pkl.dump(note_icd_graph,f)
    print("saved to :",file)

file = "note_id_to_index.pkl"
with open(file,'wb') as f:
    pkl.dump(note_id_to_index,f)
    print("saved to :",file)

