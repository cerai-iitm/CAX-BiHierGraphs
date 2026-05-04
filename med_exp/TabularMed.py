import pickle as pkl
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# get embeddings
note_embeddings = pkl.load(open('med_exp/graph/embeddings/note_embeddings.pkl', 'rb'))
icd_embedddings = pkl.load(open('med_exp/graph/embeddings/icd_embeddings.pkl', 'rb'))
num_icds = icd_embedddings.size(0)
print(note_embeddings.size(), icd_embedddings.size())

# get note - icds mapping
df = pd.read_csv("med_exp/data/notes.csv",usecols=["note_id","unique_icd_codes"])
# print(df.head())

# get (note id to index) and (icd id to index) mapping
note_id2idx = pkl.load(open('med_exp/graph/construction/note_id_to_index.pkl', 'rb'))
icd_id2idx = pkl.load(open('med_exp/graph/construction/icd_id_to_index.pkl', 'rb'))

# create a binary multilabel ground truth vector
def create_ground_truth(unique_icd_codes):
    gt = [0] * num_icds
    unique_icd_codes = unique_icd_codes.split(', ')
    for icd in unique_icd_codes:
        gt[icd_id2idx[icd]] = 1        
    return np.array(gt)

# add new columns to the dataframe with the note and icd indices
df['unique_icd_indices'] = df['unique_icd_codes'].apply(lambda x : create_ground_truth(x))
df['note_index'] = df['note_id'].apply(lambda x : note_id2idx[x])

# Define the PyTorch multilabel dataset
class MultilabelCustomDataset(Dataset):
  def __init__(self, df, embeddings):
    self.df = df
    self.embeddings = embeddings

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    X = self.embeddings[idx]
    y = torch.from_numpy(self.df['unique_icd_indices'][idx]).to(torch.float32).to(device)
    return X, y

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.activation = nn.ReLU()
        
    def forward(self, x):
        return self.activation(x + self.block(x))

class TabularMed(nn.Module):
    def __init__(self, input_dims, hidden_dim, output_dim, dropout, num_blocks=3):
        super().__init__()
        self.layer1 = nn.Linear(input_dims, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        self.output = nn.Linear(hidden_dim, output_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        z = self.layer1(x)
        z = self.norm(z)
        z = self.activation(z)
        z = self.dropout(z)
        z = self.residual_blocks(z)
        z = self.output(z)
        return z

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification (ICCV 2021).
    Paper: https://arxiv.org/abs/2009.14119
    
    Specifically designed for multilabel with many negatives.
    - gamma_neg: focusing parameter for negative samples (default: 4)
    - gamma_pos: focusing parameter for positive samples (default: 1)  
    - clip: probability margin for negatives (default: 0.05)
    - eps: label smoothing (default: 0.1)
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=0.1, 
                 disable_torch_grad_focal_loss=True):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x, y):
        """"
        Args:
            x: logits (before sigmoid), shape [batch_size, num_classes]
            y: multi-hot labels, shape [batch_size, num_classes]
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=1e-8))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()

if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 128
    HIDDEN_DIMS = 2048
    NUM_EPOCHS = 150
    LR = 5e-4
    LR_SCHEDULE = [10,]
    GAMMA = 0.1
    WEIGHT_DECAY = 1e-4  # L2 regularization

    # create datasets
    dataset = MultilabelCustomDataset(df, note_embeddings)
    # Split dataset and create dataloaders
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [0.9, 0.1], generator=generator)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, shuffle=False)
    del dataset, train_dataset, test_dataset
    torch.cuda.empty_cache()

    # init NN, loss, optimizer and LR scheduler
    tabular_med = TabularMed(note_embeddings.size(1), HIDDEN_DIMS, num_icds, 0.3).to(device)
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05, eps=0.1)
    optimizer = torch.optim.AdamW(tabular_med.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_SCHEDULE, gamma=GAMMA)

    def train(dataloader, epoch):
        tabular_med.train()
        epoch_loss = 0

        with tqdm(total=len(dataloader.dataset), desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}') as pbar:
            for X, y in dataloader:
                y_pred = tabular_med(X)
                loss = loss_fn(y_pred, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item() / X.shape[0]})

            epoch_loss /= len(dataloader.dataset)
            pbar.set_postfix(**{'loss': epoch_loss})
        
        # torch.cuda.empty_cache()
        
        return epoch_loss

    def test(dataloader):
        tabular_med.eval()
        loss = 0
        
        pred, gt, pred_tensor, gt_tensor = [], [], torch.tensor([]).to(device), torch.tensor([]).to(device)

        for X, y in dataloader:
            y_pred = tabular_med(X)
            loss += loss_fn(y_pred, y)
            # loss += loss_fn(y_pred, y).mean()
            
            pred.append(y_pred)
            gt.append(y)

        for y_pred in pred:
            pred_tensor = torch.cat((pred_tensor, y_pred), 0)
        del pred
        for y in gt:
            gt_tensor = torch.cat((gt_tensor, y), 0)
        del gt

        # torch.cuda.empty_cache()

        return loss / len(dataloader.dataset), pred_tensor, gt_tensor

    # train the model
    train_losses = []
    # for epoch in range(NUM_EPOCHS):
    #     loss = train(train_dataloader, epoch)
    #     train_losses.append(loss)
    # torch.save(tabular_med.state_dict(), "med_exp/graph/results/TabularMed.pth")

    # test the model
    tabular_med = TabularMed(note_embeddings.size(1), HIDDEN_DIMS, num_icds, 0.3).to(device)
    tabular_med.load_state_dict(torch.load("med_exp/graph/results/TabularMed.pth"))
    tabular_med.eval()
    test_loss, pred_tensor, gt_tensor = test(test_dataloader)
    probs = torch.sigmoid(pred_tensor)
    predictions = (probs > 0.55).float()

    f1_macro = MultilabelF1Score(num_icds, average="macro", threshold=0.1).to(device)
    f1_micro = MultilabelF1Score(num_icds, average="micro", threshold=0.1).to(device)
    auc = MultilabelAUROC(num_icds)

    print("Macro-F1:\t\t", f1_macro(predictions, gt_tensor).item())
    print("Micro-F1:\t\t", f1_micro(predictions, gt_tensor).item())
    print("AUC:\t\t", torch.mean(auc(pred_tensor, gt_tensor.to(torch.int))).item())


        

