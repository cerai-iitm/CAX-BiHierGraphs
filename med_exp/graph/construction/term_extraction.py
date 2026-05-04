import pandas as pd
import pickle as pkl
import ast
from tqdm import tqdm


file = "icd_id_to_index.pkl"
with open(file,'rb') as f:
    icd_id_to_index = pkl.load(f)
file = "note_id_to_index.pkl"
with open(file,'rb') as f:
    note_id_to_index = pkl.load(f)

# Note: remove the argument nrows
df1 = pd.read_csv("../../ner_data/icd10_descriptions_with_ner.csv", encoding='latin-1')

# Note: remove the argument nrows
df2 = pd.read_csv("../../ner_data/notes_title_entities.csv",usecols=["note_id","Chief Complaint_entities","Major Surgical or Invasive Procedure_entities","History of Present Illness_entities","Past Medical History_entities","Social History_entities","Family History_entities","Physical Exam_entities","Brief Hospital Course_entities","Medications on Admission_entities","Discharge Medications_entities","Discharge Disposition_entities","Discharge Diagnosis_entities","Discharge Condition_entities","Discharge Instructions_entities","Followup Instructions_entities","Pertinent Results_entities","Consolidated_Entities"]
                  )

term_set = set()

term_col_list=["BIOLOGICAL_STRUCTURE","DIAGNOSTIC_PROCEDURE","DISEASE_DISORDER","HISTORY","MEDICATION","SEVERITY","SIGN_SYMPTOM","THERAPEUTIC_PROCEDURE"]


for i,j in tqdm(df1.iterrows()):
    for col in term_col_list:
        if(isinstance(j[col], str)):
            terms = j[col].split(",")
            terms = [col+'_'+term.strip() for term in terms]
            terms = set(terms)
            term_set.update(terms)
            

print("terms from icd (unique)",len(term_set))

# term_col_list = ["Chief Complaint_entities","Major Surgical or Invasive Procedure_entities","History of Present Illness_entities","Past Medical History_entities","Social History_entities","Family History_entities","Physical Exam_entities","Brief Hospital Course_entities","Medications on Admission_entities","Discharge Medications_entities","Discharge Disposition_entities","Discharge Diagnosis_entities","Discharge Condition_entities","Discharge Instructions_entities","Followup Instructions_entities","Pertinent Results_entities","Consolidated_Entities"]

# k=0
# for i,j in tqdm(df2.iterrows()):
#     for col in term_col_list:
#         if(isinstance(j[col], str)):
#             dic = ast.literal_eval(j[col])
#             for key,val in dic.items():
#                 if len(val):
#                     terms = [key+'_'+ term for term in val]
#                     term_set.update(terms)
#                     k+=len(val)

# print("terms from notes",k)
# print("terms from both (unique)",len(term_set))


term_index={}
term_index["term_for_index"]={}
term_index["term_rev_index"]={}
index=0

for term in term_set:
    term_index["term_for_index"][term] = index
    term_index["term_rev_index"][index] = term
    index+=1

note_term_graph={}
note_term_graph["note_term_edges"]=[]

icd_term_graph={}
icd_term_graph["icd_term_edges"]=[]


term_col_list=["BIOLOGICAL_STRUCTURE","DIAGNOSTIC_PROCEDURE","DISEASE_DISORDER","HISTORY","MEDICATION","SEVERITY","SIGN_SYMPTOM","THERAPEUTIC_PROCEDURE"]

k=0
for i,j in tqdm(df1.iterrows()):
    for col in term_col_list:
        if(isinstance(j[col], str)):
            terms = j[col].split(",")
            terms = [col+'_'+term.strip() for term in terms]
            for term in terms:
                icd_term_graph["icd_term_edges"].append((icd_id_to_index[j['icd_code']], term_index["term_for_index"][term]))
                k+=1

print("terms icd edges",k)

term_col_list = ["Chief Complaint_entities","Major Surgical or Invasive Procedure_entities","History of Present Illness_entities","Past Medical History_entities","Social History_entities","Family History_entities","Physical Exam_entities","Brief Hospital Course_entities","Medications on Admission_entities","Discharge Medications_entities","Discharge Disposition_entities","Discharge Diagnosis_entities","Discharge Condition_entities","Discharge Instructions_entities","Followup Instructions_entities","Pertinent Results_entities","Consolidated_Entities"]

k=0
for i,j in tqdm(df2.iterrows()):
    for col in term_col_list:
        if(isinstance(j[col], str)):
            dic = ast.literal_eval(j[col])
            for key,val in dic.items():
                if len(val):
                    terms = [key+'_'+ term for term in val]
                    for term in terms:
                        if term in term_set:
                            note_term_graph["note_term_edges"].append((note_id_to_index[j['note_id']], term_index["term_for_index"][term]))
                            k+=1
print("note icd edges",k)

file = "term_index.pkl"
with open(file,'wb') as f:
    pkl.dump(term_index,f)
    print("saved to :",file)

file = "note_term_graph.pkl"
with open(file,'wb') as f:
    pkl.dump(note_term_graph,f)
    print("saved to :",file)

file = "icd_term_graph.pkl"
with open(file,'wb') as f:
    pkl.dump(icd_term_graph,f)
    print("saved to :",file)