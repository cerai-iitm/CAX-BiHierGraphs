import pandas as pd
import pickle as pkl

# Note: remove the argument nrows
df = pd.read_csv("../../data/icd10descriptions.csv",usecols=['icd_code'], encoding='latin-1')

icd_id_to_index={}

for i,j in df.iterrows():
    icd_id_to_index[j['icd_code']] = i

file = "icd_id_to_index.pkl"
with open(file,'wb') as f:
    pkl.dump(icd_id_to_index,f)
    print("saved to :",file)