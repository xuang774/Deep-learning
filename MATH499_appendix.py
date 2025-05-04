# Clone the IGMTF repo, extract the data, and create a model folder
!git clone https://github.com/Wentao-Xu/IGMTF.git
!cd IGMTF && tar -zxvf data.tar.gz && mkdir model

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt

file_path = '/content/IGMTF/data/exchange_rate.txt'
df = pd.read_csv(file_path, delimiter=",", header=None)  
print("Dataset preview:")
print(df.head())
print("Shape:", df.shape)

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window_size, extra_step, mode='train', normalize=True):

        # Convert to NumPy if input is a DataFrame
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        total_len = len(data)
        train_end = int(total_len * 0.6)   
        val_end = int(total_len * 0.8)    

        # Compute maximum value for each variable on the training set for normalization
        self.train_max = np.max(data[:train_end], axis=0).astype(np.float32)

        # Select appropriate data slice based on mode
        if mode == 'train':
            self.data = data[:train_end].astype(np.float32)
        elif mode == 'val':
            self.data = data[train_end:val_end].astype(np.float32)
        elif mode == 'test':
            self.data = data[val_end:].astype(np.float32)
        else:
            raise ValueError("mode should be 'train', 'val', or 'test'")

        # Optionally normalize by dividing by training max per variable
        if normalize:
            self.data = self.data / self.train_max

        self.window_size = window_size
        self.extra_step = extra_step

    def __len__(self):
        # Number of sliding windows available in the dataset
        return len(self.data) - self.window_size - self.extra_step

    def __getitem__(self, idx):
        # Get input window and corresponding label (extra_step steps ahead)
        x = self.data[idx : idx + self.window_size]  # shape: (window_size, num_features)
        y = self.data[idx + self.window_size + self.extra_step]  # shape: (num_features,)
        return torch.from_numpy(x), torch.from_numpy(y)

window_size = 168 

train_dataset = SlidingWindowDataset(df, window_size=window_size, extra_step=2, mode='train')
val_dataset   = SlidingWindowDataset(df, window_size=window_size, extra_step=2, mode='val')
test_dataset  = SlidingWindowDataset(df, window_size=window_size, extra_step=2, mode='test')

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)   # Shuffle training data
val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)    # No shuffle for validation
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)   # No shuffle for testing

class IGMTFModel(nn.Module):
    def __init__(self, d_feat=8, hidden_size=512, num_layers=2, dropout=0.0, base_model="GRU"):
        """
        Args:
            d_feat (int): Number of features per time step (only used for LSTM)
            hidden_size (int): Hidden dimension for RNN and MLP
            num_layers (int): Number of RNN layers
            dropout (float): Dropout rate inside RNN
            base_model (str): One of 'GRU' or 'LSTM'
        """
        super().__init__()

        # Choose GRU or LSTM as backbone
        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=1,  # Only one variable is used per forward pass
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("Unknown base model name %s" % base_model)

        # A 2-layer feedforward network after RNN
        self.lins = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )

        # Project embeddings for graph attention
        self.project1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.project2 = nn.Linear(hidden_size, hidden_size, bias=False)

        # Output layer after graph aggregation + direct encoding
        self.fc_out_pred = nn.Linear(hidden_size * 2, 1)

        self.leaky_relu = nn.LeakyReLU()
        self.d_feat = d_feat
    def cal_cos_similarity(self, x, y):
        """
        Compute cosine similarity between all rows of x and y.
        Args:
            x: (n, d), y: (m, d)
        Returns:
            (n, m) cosine similarity matrix
        """
        xy = x.mm(torch.t(y))  # Dot product
        x_norm = torch.sqrt(torch.sum(x * x, dim=1)).reshape(-1, 1)
        y_norm = torch.sqrt(torch.sum(y * y, dim=1)).reshape(-1, 1)
        return xy / (x_norm.mm(torch.t(y_norm)) + 1e-6)

    def sparse_dense_mul(self, s, d):
        """
        Multiply sparse tensor s with dense tensor d (only on non-zero positions of s).
        """
        i = s._indices()
        v = s._values()
        dv = d[i[0, :], i[1, :]]
        return torch.sparse.FloatTensor(i, v * dv, s.size())

    def forward(self, x, get_hidden=False, train_hidden=None, train_hidden_day=None, k_day=10, n_neighbor=5):
        device = x.device

        # Transpose input: (1, 168, F) → (F, 168, 1)
        x = x.permute(2, 1, 0)
        out, _ = self.rnn(x)  # Shape: (F, T, H)
        out = out[:, -1, :]  # Shape: (F, H)
        out = self.lins(out)  # Shape: (F, H)
        mini_batch_out = out  # Feature-wise hidden encodings
        if get_hidden:
            return mini_batch_out
        # Average across features to represent "the day"
        mini_batch_out_day = torch.mean(mini_batch_out, dim=0).unsqueeze(0)
        
        # Compute cosine similarity to all stored day-level hidden states
        day_similarity = self.cal_cos_similarity(mini_batch_out_day, train_hidden_day.to(device))
        
        # Get top-k most similar days from training memory
        day_index = torch.topk(day_similarity, k_day, dim=1)[1]
       
        # Retrieve hidden states for those days
        sample_train_hidden = train_hidden[day_index.long().cpu()].squeeze()
        sample_train_hidden = [torch.from_numpy(h.astype(np.float32)).to(device) for h in sample_train_hidden]
        sample_train_hidden = torch.cat(sample_train_hidden, dim=0)  # Shape: (k_day * F, H)
        sample_train_hidden = self.lins(sample_train_hidden)  # Refine memory encodings
        
        # Compute pairwise cosine similarity between query and memory
        cos_similarity = self.cal_cos_similarity(self.project1(mini_batch_out), self.project2(sample_train_hidden))
        
        # Build sparse top-k graph connection matrix (attention weights)
        row = (
            torch.arange(x.shape[0]).reshape([-1, 1])
            .repeat(1, n_neighbor)
            .reshape(1, -1)
            .to(device)
        )
        column = torch.topk(cos_similarity, n_neighbor, dim=1)[1].reshape(1, -1)

        mask = torch.sparse_coo_tensor(
            torch.cat([row, column]),
            torch.ones([row.shape[1]]).to(device) / n_neighbor,
            (x.shape[0], sample_train_hidden.shape[0]),
        )

        # Apply attention weights to the memory states
        cos_similarity = self.sparse_dense_mul(mask, cos_similarity)
        agg_out = torch.sparse.mm(cos_similarity, self.project2(sample_train_hidden))  # Shape: (F, H)

        # Concatenate direct encoding + neighbor encoding → final prediction
        out = self.fc_out_pred(torch.cat([mini_batch_out, agg_out], axis=1)).squeeze()  # Shape: (F,)

        return out

def get_train_hidden(train_loader, igmtf_model):
    """
    Precompute and cache hidden representations from the training set.

    Returns:
        train_hidden (list): List of instance-level hidden encodings
        train_hidden_day (Tensor): Averaged hidden states per day
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    igmtf_model.eval()

    train_hidden = []
    train_hidden_day = []

    for daily_feature, _ in train_loader:
        daily_feature = daily_feature.to(device)
        out = igmtf_model(daily_feature, get_hidden=True)  # Shape: (F, H)

        # Save feature-wise encodings (as numpy object for flexibility)
        train_hidden.append(out.detach().cpu())

        # Save day-level average encoding: (1, H)
        train_hidden_day.append(out.detach().cpu().mean(dim=0).unsqueeze(0))

    # List of (F, H) tensors → NumPy objects
    train_hidden = np.asarray(train_hidden, dtype=object)

    # Stack day encodings: (T_days, H)
    train_hidden_day = torch.cat(train_hidden_day)

    return train_hidden, train_hidden_day

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = IGMTFModel().to(device)
train_hidden, train_hidden_day = get_train_hidden(train_loader, model)

def validation_process(val_loader, train_hidden, train_hidden_day, model):
    """
    Evaluate model performance on validation set using RSE (Relative Squared Error).
    
    Args:
        val_loader (DataLoader): Validation data loader
        train_hidden: Cached hidden states of training instances
        train_hidden_day: Cached day-level hidden averages
        model (IGMTFModel): Trained IGMTF model

    Returns:
        rse (float): Relative Squared Error on validation set
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    numerator = 0.0     # ∑ (y - ŷ)^2
    denominator = 0.0   # ∑ (y - ȳ)^2
    all_targets = []    # Collect all targets to compute mean(y)

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x, train_hidden=train_hidden, train_hidden_day=train_hidden_day)

            numerator += torch.sum((y - pred) ** 2).item()
            all_targets.append(y.squeeze())

    all_targets = torch.cat(all_targets, dim=0)
    mean_y = torch.mean(all_targets)
    denominator = torch.sum((all_targets - mean_y) ** 2).item()

    rse = (numerator ** 0.5) / (denominator ** 0.5)  # Compute RSE
    return rse
def train_epoch(train_dataloader, train_hidden, train_hidden_day, val_loader, igmtf_model, total_epoch=100):
 
    loss_fn = nn.MSELoss()
    train_optimizer = optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    igmtf_model = igmtf_model.to(device)

    train_loss_record = {'train': []}
    val_loss_record = {'RSE': []}

    # --- Early stopping setup ---
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None

    # --- Setup for feature-splitting ---
    perm = None       
    iter = 0          
    step_size = 100   
    num_split = 3     

    for epoch in range(total_epoch):
        igmtf_model.train()
        print(f'Epoch: {epoch}')
        batch_loss = []

        for feature_daily, label_daily in train_dataloader:
            train_optimizer.zero_grad()

            feature_daily = feature_daily.to(device)
            label_daily = label_daily.to(device)
            B, T, F = feature_daily.shape

            # Update feature permutation every step_size iterations
            if iter % step_size == 0:
                perm = torch.randperm(F)
                print(f"🔁 Updated permutation at iteration {iter}: {perm.tolist()}")

            total_loss = 0.0
            num_sub = int(F / num_split)

            # --- Train on split subsets of features ---
            for j in range(num_split):
                if j != num_split - 1:
                    idx = perm[j * num_sub : (j + 1) * num_sub]
                else:
                    idx = perm[j * num_sub :]

                tx = feature_daily[:, :, idx]  # (B, T, sub_features)
                ty = label_daily[:, idx]       # (B, sub_features)
                pred = igmtf_model(tx, train_hidden=train_hidden, train_hidden_day=train_hidden_day)
                loss = loss_fn(pred, ty)
                total_loss += loss * tx.shape[2]  # Weight by number of features in group

            train_batch_loss = total_loss / F  # Normalize over total number of features
            train_batch_loss.backward()

            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_value_(igmtf_model.parameters(), 5.0)
            train_optimizer.step()

            train_loss_record['train'].append(train_batch_loss.detach().cpu().item())
            iter += 1  # Update iteration counter

        # --- Validation ---
        val_loss = validation_process(
            val_loader,
            train_hidden=train_hidden,
            train_hidden_day=train_hidden_day,
            model=igmtf_model
        )
        print('Val loss:', val_loss)
        val_loss_record['RSE'].append(val_loss)

        # --- Early stopping check ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = igmtf_model.state_dict()
            print(f"✅ Validation loss improved. Saving model at epoch {epoch}.")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement. Early stop counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("⏹️ Early stopping triggered!")
            break

    # --- Save best model ---
    if best_model_state:
        torch.save(best_model_state, '/content/igmtf_train_model')
        print("✅ Best model saved.")
    else:
        torch.save(igmtf_model.state_dict(), '/content/igmtf_train_model')
        print("✅ Last model saved.")

    return train_loss_record, val_loss_record

# Start training
train_loss_record, val_loss_record = train_epoch(
    train_loader,
    train_hidden,
    train_hidden_day,
    val_loader,
    model,
    total_epoch=100
)

def test_process(test_loader, train_hidden, train_hidden_day, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)

    numerator = 0.0               # Sum of squared errors (for RSE numerator)
    all_targets = []              # List to collect all ground truth values
    all_preds = []                # List to collect all predicted values

    with torch.no_grad():       
        for x, y in test_loader:
            x = x.to(device)                   
            y = y.to(device).squeeze()        
            pred = model(x, train_hidden=train_hidden, train_hidden_day=train_hidden_day)  # Get model prediction

            # Accumulate squared error for RSE numerator
            numerator += torch.sum((y - pred) ** 2).item()

            # Store predictions and targets for later use (e.g., CORR, RSE denominator)
            all_targets.append(y)
            all_preds.append(pred)

    # Concatenate all targets and predictions along the time dimension
    all_targets = torch.cat(all_targets, dim=0)  # Shape: (T, n)
    all_preds = torch.cat(all_preds, dim=0)      # Shape: (T, n)

    # Compute RSE denominator: variance of ground truth
    mean_y = torch.mean(all_targets)
    denominator = torch.sum((all_targets - mean_y) ** 2).item()

    # Final RSE computation (add epsilon to prevent division by zero)
    rse = (numerator ** 0.5) / (denominator ** 0.5 + 1e-8)

    # === CORR: compute average Pearson correlation coefficient over all variables ===
    n = all_targets.shape[1]  # Number of variables
    corr_sum = 0.0

    for i in range(n):
        yi = all_targets[:, i]     
        yhat_i = all_preds[:, i]   
        mean_yi = yi.mean()
        mean_yhat_i = yhat_i.mean()

        # Compute Pearson correlation for variable i
        num = torch.sum((yi - mean_yi) * (yhat_i - mean_yhat_i))
        denom = torch.sqrt(torch.sum((yi - mean_yi) ** 2)) * torch.sqrt(torch.sum((yhat_i - mean_yhat_i) ** 2))

        corr_i = num / (denom + 1e-8)
        corr_sum += corr_i.item()

    # Average correlation over all variables
    corr = corr_sum / n

    return rse, corr

model.load_state_dict(torch.load('/content/igmtf_train_model'))
rse, corr = test_process(test_loader, train_hidden, train_hidden_day, model)
print(f"✅ Test RSE: {rse:.4f}")
print(f"✅ Test CORR: {corr:.4f}")
