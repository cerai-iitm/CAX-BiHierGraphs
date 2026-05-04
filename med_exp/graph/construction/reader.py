import pandas as pd
import pickle as pkl
note_file = "note_id_to_index.pkl"
with open(note_file,'rb') as f:
    icd_id_to_index = pkl.load(f)

print(len(icd_id_to_index.keys()))