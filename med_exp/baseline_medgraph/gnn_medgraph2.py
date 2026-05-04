import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch_geometric
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import math
import torch_geometric.transforms as T
from sklearn.metrics import classification_report
from torch_geometric.nn import GAT
from torchmetrics import Accuracy, F1Score
import pickle
import matplotlib.pyplot as plt
import seaborn as sn
from torchmetrics.classification import MultilabelConfusionMatrix
from torchmetrics.classification import MultilabelConfusionMatrix, MultilabelAUROC # Added AUROC
# Setup
print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print('Starting the process!')
torch_geometric.seed_everything(1)
torch.manual_seed(1)
EPOCS = 201



# Metrics
acc = Accuracy(task="multilabel", num_labels=9000, average='weighted').to(device)
f1 = F1Score(task="multilabel", num_labels=9000, average='weighted').to(device)
auc_metric = MultilabelAUROC(num_labels=9000, average='weighted').to(device)
# 1. Load Data
print("Loading Data...")
with open('./med_exp/medgraph_exp/MedGraph.pkl', 'rb') as f:
    # Assuming pickle contains a Data object or dict. 
    # If it is a dict, we convert it immediately.
    data = pickle.load(f)
    from torch_geometric.utils import dropout_edge
    edge_index, _ = dropout_edge(data.edge_index, p=0.8)

    data.edge_index = edge_index


# 2. Split Data using RandomNodeSplit
# print("Splitting data using RandomNodeSplit...")
# transform = T.RandomNodeSplit(
#     num_val=0.3,
#     num_test=0.3,
#     # split='train_rest' # Optional: define how the rest are handled
# )

# data = transform(data)
# print(data)
# # Ensure types
# if hasattr(data, 'edge_weight') and data.edge_weight is not None:
#     data.edge_weight = data.edge_weight.float()
# if hasattr(data, 'edge_attr') and data.edge_attr is not None:
#     data.edge_attr = data.edge_attr.float()

# # Move single data object to GPU
# data = data.to(device)

# print(f"Data Info: {data}")
# print(f"Train mask sum: {data.train_mask.sum().item()}")
# print(f"Val mask sum: {data.val_mask.sum().item()}")
# print(f"Test mask sum: {data.test_mask.sum().item()}")

num_nodes = data.num_nodes
perm = torch.randperm(num_nodes)

train_idx = perm[:int(0.7 * num_nodes)]
val_idx   = perm[int(0.7 * num_nodes):int(0.85 * num_nodes)]
test_idx  = perm[int(0.85 * num_nodes):]

# 2. Extract induced subgraphs
# Each subgraph only contains edges where BOTH nodes are in the index list.
train_data = data.subgraph(train_idx)
val_data   = data.subgraph(val_idx)
test_data  = data.subgraph(test_idx)

train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)




# 3. Model Setup
model = GAT(
    in_channels=data.num_features, 
    hidden_channels=150, 
    num_layers=2, 
    out_channels=9000,
    v2=True, 
    edge_dim=2, 
    # return_attention_weights=True
)

model.cuda()
print(model(train_data.x, train_data.edge_index, edge_attr=train_data.edge_attr).shape)
# 4. Calculate Weights (Vectorized)
print("Calculating class weights...")
# Only calculate weights based on Training Data to avoid leakage
train_labels = train_data.y
pos_counts = train_labels.sum(dim=0)
neg_counts = (train_labels == 0).sum(dim=0)

# Protect against division by zero
pos_counts = torch.where(pos_counts == 0, torch.ones_like(pos_counts), pos_counts)

ratio = neg_counts / pos_counts
class_weights = (torch.sqrt(ratio) * 1.5).float().to(device)
print(f"Class Weights: {class_weights}")

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)


# 5. Helper Functions
def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename))

def train():
    model.train()
    optimizer.zero_grad()
    
    # GAT takes the whole graph structure
    out = model(train_data.x, train_data.edge_index, edge_attr=train_data.edge_attr)
    
    # We only compute loss on nodes in the TRAINING MASK
    out_train = out
    y_train = train_data.y.float()
    
    loss = loss_fn(out_train, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(data):
    model.eval()
    out = model(data.x, data.edge_index, edge_attr=data.edge_attr)
    
    # Filter by mask (Val or Test)
    out_masked = out
    y_masked = data.y.float()
    
    # Loss
    val_loss = loss_fn(out_masked, y_masked)
    
    # Metrics
    pred_logits = F.sigmoid(out_masked)
    pred_labels = pred_logits.round()
    
    accuracy = acc(pred_labels, y_masked.int())
    f1_score = f1(pred_labels, y_masked.int())
    # New: AUC calculation
    auc_score = auc_metric(pred_logits, y_masked.int())
    return val_loss.item(), accuracy.item(), f1_score.item(), auc_score.item(), pred_labels, y_masked


# 6. Training Loop
val_list_acc = []
val_list_f1 = []
loss_list = []
best_val_f1 = -1

print("Starting training...")
for epoch in tqdm(range(1, EPOCS)):
    train_loss = train()
    loss_list.append(train_loss)
    
    # Validation
    val_loss, val_acc, val_f1,val_auc, _, _ = test(val_data)
    
    val_list_acc.append(val_acc)
    val_list_f1.append(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        checkpoint(model, 'best_model_checkpoint_GAT_sent_attr_meld.pth')
    
    # Optional: Print less frequently to save console space
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Val F1: {val_f1:.4f}')

# 7. Final Testing
print("Loading best model for testing...")
resume(model, 'best_model_checkpoint_GAT_sent_attr_meld.pth')

test_loss, test_acc, test_f1,test_auc, test_pred, test_true = test(test_data)

print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test F1-score: {test_f1:.4f}')
print(f'Test AUC: {test_auc:.4f}')
# Save Results
os.makedirs('./med_exp/medgraph_exp/results/', exist_ok=True)
with open('./med_exp/medgraph_exp/results/final_results_GAT_sent_attr.txt', 'a') as f:
    f.write(f'\nTest Accuracy: {test_acc:.4f}')
    f.write(f'Test F1-score: {test_f1:.4f}')
    f.write(f'\nTest AUC: {test_auc:.4f}')

# 8. Visualization
print("Generating Plots...")

# Confusion Matrix
cm = MultilabelConfusionMatrix(num_labels=9000).to(device)
# Ensure predictions and targets are on same device and correct type
cm_out = cm(test_pred, test_true.int())
x = cm_out.cpu()

f, ax = plt.subplots(5, 6, figsize=(20, 20))
# Flatten ax for easy iteration if necessary, or loop carefully
ax = ax.flatten()

for i in range(9000):
    if i < len(ax):
        sn.heatmap(x[i], annot=True, fmt='.0f', cbar=False, ax=ax[i])
        ax[i].set_title(f'Class {i}')

plt.tight_layout()
f.savefig("./med_exp/medgraph_exp/results/Confusion_Matrix_GAT_multilabel_wts.jpg")
plt.close(f)

# Loss/Metric Curves
plt.figure(figsize=(25,15))
time_epoch = range(1, EPOCS)
plt.plot(time_epoch, loss_list, label='Train Loss', color='b')
plt.plot(time_epoch, val_list_acc, label='Val Acc', color='g')
plt.plot(time_epoch, val_list_f1, label='Val F1', color='r')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()

ax = plt.gca()
from matplotlib.ticker import MultipleLocator
ax.xaxis.set_major_locator(MultipleLocator(10))

plt.savefig("./med_exp/medgraph_exp/results/Fit_Graph_GAT_mlb_wts.jpg")
plt.close()

torch.cuda.empty_cache()
print("Process Completed.")