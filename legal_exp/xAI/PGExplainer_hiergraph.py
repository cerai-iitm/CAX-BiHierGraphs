import sys
import pickle, os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch

device = torch.device('cuda') if (torch.cuda.is_available()) else torch.device('cpu') 
print(device)

from gnn_hier_graph import Model
import pickle
from tqdm import tqdm
from HeteroPGExplainer import HeteroPGExplainer
import torch_geometric.transforms as T
from torch_geometric import seed_everything
import pickle as pkl

torch.set_num_threads(4)
torch.set_num_interop_threads(4)



#seeding everything with same key to maintain same data split across both link pred model and explainer model
seed_everything(4321)

hier_graph = pickle.load(open('datasets/Hetero_Data_With_Self_Loops.pkl', 'rb'))
case_embeds_legal_bert = pickle.load(open('../LegalGraph/code/dumps/legalbert_embeds_train.pkl', 'rb'))
case_embeds_legal_bert = torch.cat(case_embeds_legal_bert, dim=0)

# Init/Retrieve link pred Model
model = Model(hier_graph, case_embeds_legal_bert, hier_graph['articles'].x.shape[1], 64)
model.to(device)
model.load_state_dict(torch.load('results/hetero_gnn_model.pt'))
model.eval()



#We use the test data after RandomLinkSplit
transform = T.RandomLinkSplit(
                    num_val=0.2,
                    num_test=0.2,  
                    disjoint_train_ratio=0.3,
                    neg_sampling_ratio=2.0,
                    edge_types=("cases", "violate", "articles"),
                    rev_edge_types=("articles", "rev_violate", "cases"),)
train_data, val_data, test_data = transform(hier_graph)
del hier_graph,train_data,val_data


#Create instance of HeteroPGExplainer
pgexplainer = HeteroPGExplainer(model, 
                                num_hops=3, 
                                ghetero=test_data,
                                lr=0.005,
                                alpha1=1e-2, 
                                alpha2=5e-4, 
                                in_dim=64, 
                                K=2,
                                mask_generator_hidden_dim=64,
                                num_epochs=100,device = device).to(device)

Task = input("\nEnter t for training model \nEnter e for explaining\n")


explainer_model_save_path = './results/explanations/hetero_pg_explainer/hetero_pg_explainer.pt'
results_path = './results/explanations/hetero_pg_explainer'
explanations_path = './results/explanations/hetero_pg_explainer'


test_data.to(device)
#train the explainer model
if Task == 't':
    print("\nTraining\n")

    pgexplainer.train_mask_generator(test_data,("cases", "violate", "articles"),device,batch_size = 72)

    torch.save(pgexplainer.state_dict(), explainer_model_save_path)
    with open(os.path.join(results_path, 'epochwise_losses.pkl'), 'wb') as file:
        pkl.dump(pgexplainer.epoch_loss, file)
# Explain all the test edges with the trained model
elif Task == 'e':
    print("\Explaining\n")

    test_edges = test_data['cases', 'violate', 'articles'].edge_label_index
    pgexplainer.load_state_dict(torch.load(explainer_model_save_path))
    pred_mask={}

    for i in range(1):
    # for i in tqdm(range(test_edges.shape[1])):
        with torch.no_grad():
            test_data["cases", "violate", "articles"].edge_label_index = torch.stack((test_edges[0][i],test_edges[1][i]))
            temp = pgexplainer.explain(test_data,("cases", "violate", "articles"),device)
        pred_mask[(test_edges[0][i].item(),test_edges[1][i].item())]={}
        for k,v in temp.items():
            pred_mask[(test_edges[0][i].item(),test_edges[1][i].item())][k] = v.detach().cpu()
        del temp
            # print(sys.getsizeof(pred_mask))
    # with open(os.path.join(explanations_path, 'explanations.pkl'), 'wb') as file:
    #     pkl.dump(pred_mask, file)
    with open(os.path.join(explanations_path, 'explanations_one.pkl'), 'wb') as file:
        pkl.dump(pred_mask, file)




