
import pickle as pkl
from datasets import load_dataset


def main():

    
    #################################
    ## Getting facts from ECthr
    #################################
    all_cases_data = load_dataset('ecthr_cases', 'alleged-violation-prediction')
    df_cases = all_cases_data["train"].to_pandas()
    
    n = len(df_cases)
    case_fact_edges = []
    fact_num =0
    k=0
    for case_num in range(n):
        facts = df_cases.loc[case_num, 'facts']
        for _ in range(len(facts)):
            case_fact_edges.append([fact_num, case_num])
            if(len(facts[_])>512):
                k+=1
            
            fact_num+=1
    case_fact_graph = {'case_fact_edges' :case_fact_edges }
    print(fact_num,k)
    with open('datasets/case_fact_graph.pkl', 'wb') as file:
        pkl.dump(case_fact_graph, file)
if __name__=='__main__':
    main()
