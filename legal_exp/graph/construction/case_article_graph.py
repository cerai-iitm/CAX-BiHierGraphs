
import pickle as pkl
from datasets import load_dataset
import pandas as pd


def main():

    
    #################################
    ## Getting data from ECthr
    #################################
    all_cases_data = load_dataset('ecthr_cases', 'violation-prediction')
    df_cases = all_cases_data["train"].to_pandas()

    article_file_path = './datasets/ECHR_Articles_new.csv'
    df_articles = pd.read_csv(article_file_path, delimiter='\t')
    mp = {}
    for i in range(len(df_articles)):
        article_id = df_articles.loc[i, 'Article Id']
        mp[article_id] = i
    # print(mp)
    
    
    case_article_edges=[]
    n = len(df_cases)
    c=0
    for case_num in range(n):
        articles_list = df_cases.loc[case_num, 'labels']
        for article_id in articles_list:
            if(article_id == 'P1-1'):
                case_article_edges.append([case_num,mp['P-1']])
            elif(article_id == 'P1-2' or article_id == 'P1-3'):
                c+=1
                continue
            else:
                case_article_edges.append([case_num,mp[article_id]])
        
    print(len(case_article_edges),c)
    
    with open('datasets/case_article_graph.pkl', 'wb') as file:
        pkl.dump(case_article_edges, file)
    
if __name__=='__main__':
    main()
