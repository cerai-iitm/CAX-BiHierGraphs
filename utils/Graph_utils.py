from datasets import load_dataset
import pickle as pkl
import pandas as pd


# File paths
article_file_path = '/home/gokul/Hier-Legal-Graph/datasets/ECHR_Articles_new.csv'
article_term_graph_path = 'datasets/article_term_graph.pkl'
case_fact_graph_file = './datasets/case_fact_graph.pkl'

# Getting all case data
all_cases_data = load_dataset('ecthr_cases', 'violation-prediction')
df_cases = all_cases_data["train"].to_pandas()

# Getting case_fact edges
with open(case_fact_graph_file, 'rb') as file:
        case_fact_graph = pkl.load(file)
    
# Getting terms from article_term_graph
with open(article_term_graph_path, 'rb') as file:
    article_term_graph = pkl.load(file)

# Getting articles
df_articles = pd.read_csv(article_file_path, delimiter='\t')

# Retriving node text from node Id
fact_case_offset = {}
offset = 0
prev_case =-1
for i in range(len(case_fact_graph['case_fact_edges'])):
    if case_fact_graph['case_fact_edges'][i][1] != prev_case:
         offset=0
    fact_case_offset[i] = [offset,case_fact_graph['case_fact_edges'][i][1]]
    prev_case = case_fact_graph['case_fact_edges'][i][1]
    offset+=1


#####################################################################################
#####################################################################################
# Functions
#####################################################################################
#####################################################################################

def getFact(node_id):
    offset = fact_case_offset[node_id][0]
    case_num = fact_case_offset[node_id][1]
    return df_cases.loc[case_num, 'facts'][offset]

def getTerm(node_id):
    return article_term_graph['term_reverse_index'][node_id]

def getArticle(node_id):
    if pd.isna(df_articles.loc[node_id, 'Title']):
         return "Title : " + "\nDescription : "+df_articles.loc[node_id, 'Description']
    return "Title : "+ df_articles.loc[node_id, 'Title'] + "\nDescription : "+df_articles.loc[node_id, 'Description']

def getCase(node_id):
    temp = f"Case_{node_id}\n"
    facts = df_cases.loc[node_id, 'facts']
    for fact in facts:
        temp += fact + '\n'
    return temp

def getNodeText(node_id, node_type):
    # Returns text of a node given nodeid and its type
    if node_type == 'facts' :
        return  getFact(node_id)
    elif node_type == 'terms':
        return getTerm(node_id)
    elif node_type == 'articles':
        return getArticle(node_id)
    elif node_type == 'cases':
        return getCase(node_id)
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

