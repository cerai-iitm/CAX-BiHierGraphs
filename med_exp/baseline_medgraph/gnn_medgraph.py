import torch, torch_geometric
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import math
import torch_geometric.transforms as T
from sklearn.metrics import classification_report
import os
from torch_geometric.nn import GAT
from torchmetrics import Accuracy, F1Score
print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

print('Starting the process!')
torch_geometric.seed_everything(1)

acc = Accuracy(task="multilabel", num_labels = 28, average='weighted').to(device)
f1 = F1Score(task="multilabel", num_labels = 28, average='weighted').to(device)

import pickle

with open('./med_exp/medgraph_exp/MedGraph.pkl', 'rb') as f:
    di = pickle.load(f)
print("Splitting data")
transform = T.RandomLinkSplit(
                    num_val=0.2,
                    num_test=0.2,  
                    disjoint_train_ratio=0.3,
                    neg_sampling_ratio=10.0,
                    # add_negative_train_samples=False, 
                    edge_types=('notes','links','icds'),
                    rev_edge_types=("icds", "rev_links", "notes"),)
train_data, val_data, test_data = transform(di)
# train_data = torch_geometric.data.Data.from_dict(di)
# print('Train graph', train_data)
#train_data.edge_weight=train_data.edge_weight.type(torch.float)
#train_data.edge_attr=train_data.edge_attr.type(torch.float)
train_data.cuda()

model = GAT(in_channels=train_data.num_features, hidden_channels=150, num_layers=2, out_channels=100, v2=True, edge_dim=2, return_attention_weights=True)
print(train_data.num_features)
model.cuda()

val_list_acc = []
val_list_f1 = []

weights_list = []


for length in tqdm(range(28)):
    i = 0
    j = 0
    k = 0
    for z in tqdm(range(9000)):
        hold = train_data.y[z]
        if hold[length] == 1:
            j+=1
        else:
            i+=1
    k=i/j
    k=(math.sqrt(k))*1.5
    weights_list.append(k)


optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
#weights_list = [3.29,1.00]
class_weights = torch.FloatTensor(weights_list).cuda()
#loss_fn = torch.nn.CrossEntropyLoss(weight = class_weights)
#loss_fn = torch.nn.CrossEntropyLoss()
#loss_fn = torch.nn.MultiLabelSoftMarginLoss()
print(class_weights)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename))

def train(data):
    
    best_val_acc = -1
    best_val_f1 = -1
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, edge_attr=data.edge_attr)
    
    loss = loss_fn(out, data.y.float())
    loss.backward()
    optimizer.step()

    val_acc, val_f1, _ = test(val_data, val=True)
    #val_list_acc.append(val_acc)
    val_list_f1.append(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        checkpoint(model, 'best_model_checkpoint_GAT_sent_attr_meld.pth')

    return loss

def test(data, val=False):
    model.eval()
    out = model(data.x, data.edge_index, edge_attr=data.edge_attr)
    out = F.sigmoid(out)
    val_loss = loss_fn(out, data.y.float())
    pred = out.round()
    val_list_acc.append(val_loss.item())
    test_acc = acc(pred, data.y)
    test_f1 = f1(pred, data.y)
    #if not val:
    #    with open('classification_report_GAT_with_sent_attr.txt', 'a') as f:
    #        f.write(str(classification_report(pred, data.y, target_names = [ 'ang', 'disg', 'fear', 'joy', 'neu', 'sad', 'sur'] )) + '\n')
    pred.cuda()
    return test_acc, test_f1, pred


# with open('./graphs/val_graph_new_new_mlb.pkl', 'rb') as f:
#     di = pickle.load(f)
val_data = torch_geometric.data.Data.from_dict(di)
#val_data.edge_weight=val_data.edge_weight.type(torch.float)
#val_data.edge_attr=val_data.edge_attr.type(torch.float)
val_data.cuda()

loss_list=[]
for epoch in tqdm(range(1, 151)):
    loss = train(train_data)
    loss_list.append(loss)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

resume(model, 'best_model_checkpoint_GAT_sent_attr_meld.pth')

# with open('../med_exp/medgraph_exp/test_graph_new_new_mlb.pkl', 'rb') as f:
#     di = pickle.load(f)
test_data = torch_geometric.data.Data.from_dict(di)
#test_data.edge_weight=test_data.edge_weight.type(torch.float)
#test_data.edge_attr=test_data.edge_attr.type(torch.float)
test_data.cuda()

test_acc, test_f1, pred = test(test_data)
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test F1-score: {test_f1:.4f}')

with open('./med_exp/medgraph_exp/results/final_results_GAT_sent_attr.txt', 'a') as f:
    f.write(f'\nTest Accuracy: {test_acc:.4f}')
    f.write(f'Test F1-score: {test_f1:.4f}')


import matplotlib.pyplot as plt
import seaborn as sn
#plt.figure(1, figsize=(9, 6))




from torchmetrics.classification import MultilabelConfusionMatrix
cm = MultilabelConfusionMatrix(num_labels=28).to(device)
cm(pred,test_data.y)
print(cm)
fig_ , ax_  = cm.plot()
x = cm(pred, test_data.y).cpu()

f, ax = plt.subplots(5, 6, figsize=(20, 20))
for i in range(28):
  sn.heatmap(x[i], annot=True, fmt='.0f', cbar=False, ax=ax[i//6][i%6])
f.savefig("./med_exp/medgraph_exp/results/Confusion_Matrix_GAT_multilabel_wts.jpg")
#fig_.savefig("Confusion_Matrix_GCN_multilabel.jpg")


loss_list = torch.tensor(loss_list)
val_list_acc = torch.tensor(val_list_acc)
val_list_f1 = torch.tensor(val_list_f1)
loss_list.cpu()
val_list_acc.cpu()
val_list_f1.cpu()
time_epoch = range(1,151)

plt.figure(figsize=(25,15))
plt.plot(time_epoch, loss_list, label = 'Train Score', color = 'b')
plt.plot(time_epoch, val_list_acc[:-1], label = 'Val Loss', color = 'g')
plt.plot(time_epoch, val_list_f1, label = 'F1', color = 'r')
plt.xlabel('Epochs')
plt.ylabel('Metrics')

plt.show()

ax = plt.gca()
n = 10
from matplotlib.ticker import MultipleLocator
ax.xaxis.set_major_locator(MultipleLocator(n))
plt.savefig("./med_exp/medgraph_exp/results/Fit_Graph_GAT_mlb_wts.jpg")

torch.cuda.empty_cache()




# cm.sum(axis=0)
 
'''
# sn.set(font_scale=0.7)
labels = ["Ang", "Disg", "Fear", "Joy", "Neu", "Sad", "Sur"]
g = sn.heatmap(np.transpose(cm/cm.sum(axis=0)*100.0), annot=True, xticklabels=labels, yticklabels=labels, annot_kws={"size": 16}, cmap="YlOrBr", fmt=".2f")

g.set_xticklabels(g.get_xticklabels(), size=18)
g.set_yticklabels(g.get_yticklabels(), size=18)

plt.yticks(rotation=0)
plt.xticks(rotation=0)

g.set_xlabel("Predicted Label", fontsize=20)
g.set_ylabel("True Label", fontsize=20)

# plt.show()
plt.savefig('confusion_matrix_MELD_GAT_sent_attr.pdf', bbox_inches='tight')
'''
