import pandas as pd
import pickle as pkl

df_icd = pd.read_csv("med_exp/data/icd10descriptions.csv", encoding='latin-1')
df_note = pd.read_csv("med_exp/data/notes.csv",usecols=["Chief Complaint","Past Medical History"])
with open('med_exp/graph/construction/term_index.pkl', 'rb') as file:
        term_index = pkl.load(file)


#####################################################################################
#####################################################################################
# Functions
#####################################################################################
#####################################################################################
def getNote(node_id):
    j = df_note.loc[node_id]
    if isinstance(j['Chief Complaint'], str) and isinstance(j['Past Medical History'], str):
        text = 'Chief Complaint : '+j['Chief Complaint'] + '\nPast Medical History : ' + j['Past Medical History']
    elif isinstance(j['Chief Complaint'], str):
        text = 'Chief Complaint : '+j['Chief Complaint']
    elif isinstance(j['Past Medical History'], str):
        text = 'Past Medical History : ' + j['Past Medical History']
    else:
        text = ""
    return text

def getTerm(node_id):
    return term_index['term_rev_index'][node_id]

def getIcd(node_id):
    j = df_icd.loc[node_id]
    return j['long_title'] + '\n' + j['Detailed Description']

def getNodeText(node_id,node_type):
    # Returns text of a node given nodeid and its type
    if node_type == 'notes' :
        return  getNote(node_id)
    elif node_type == 'terms':
        return getTerm(node_id)
    elif node_type == 'icds':
        return getIcd(node_id)
    else:
        return "INVALID NODE TYPE"

def getEdgeText(edge):
    # Assuming edge is a  tuples in format (id1,id2,edgeType)
    # Assuming edge Type is of the form ('articles', 'has', 'terms')
    # return type is a list of string in  [text1,text2]

    nodeType_1 = edge[2][0]
    nodeId_1 = edge[0]
    nodeType_2 = edge[2][2]
    nodeId_2 = edge[1]

    return [getNodeText(nodeId_1,nodeType_1),getNodeText(nodeId_2,nodeType_2)]



