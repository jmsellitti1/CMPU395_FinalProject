import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

df = pd.read_parquet('data/features.parquet')

# Features
feature_cols = [
    'AVG', 'SLG', 'OBP', 'HR', 'xwOBA',
    'K%', 'BB%', 'Contact%', 'SweetSpot%', 'HardHit%'
]

X = df[feature_cols].values
y = df['lineup_pos'].astype(int).values

# train / validation split

# Create groups to ensure all 9 players from a single game stay together in the same split
df['game_id'] = df['team'] + "_" + df['game_date'].astype(str)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups=df['game_id']))

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]
val_df = df.iloc[val_idx].copy()

# feature normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X = scaler.transform(X)

class LineupDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        ordinal = torch.zeros(8)
        ordinal[:y-1] = 1

        return x, ordinal

train_loader = DataLoader(LineupDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(LineupDataset(X_val, y_val), batch_size=32)

X_tensor = torch.tensor(X, dtype=torch.float32)

class LineupModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)

        self.out = nn.Linear(32, 1)
        self.thresholds = nn.Parameter(torch.linspace(-2, 2, 8))
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        base = self.out(x)
        logits = base - self.thresholds
        return logits

model = LineupModel(X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

num_epochs = 200
patience = 20
min_delta = 1e-3
best_val_loss = float('inf')
counter = 0
best_model_state = None

epoch_losses = {}

for epoch in tqdm(range(num_epochs), desc="Training Model"):
    # Training
    model.train()
    train_loss = 0

    for X_batch, y_batch in train_loader:
        preds = model(X_batch)
        loss = criterion(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            val_loss += loss.item()
            
    val_loss /= len(val_loader)

    # Early stopping check
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        counter = 0
        best_model_state = model.state_dict()
    else:
        counter += 1

    if counter >= patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break

    epoch_losses[epoch+1] = {
        'train_loss': train_loss,
        'val_loss': val_loss
    }

# Plot training and validation loss
epochs = list(epoch_losses.keys())
train_losses = [epoch_losses[ep]['train_loss'] for ep in epochs]
val_losses = [epoch_losses[ep]['val_loss'] for ep in epochs]

plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("data/loss_plot.png")
print("Loss plot saved to loss_plot.png")

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

def ordinal_to_class(logits):
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1) + 1

with torch.no_grad():
    logits = model(X_tensor)
    preds = ordinal_to_class(logits).numpy()

pred_df = pd.DataFrame(preds, columns=['predicted_lineup_pos'])

# Attach metadata and true labels
metadata = df[['team', 'player_id', 'game_date']].reset_index(drop=True)
true_labels = df['lineup_pos'].astype(int).values
final_df = pd.concat([metadata, pred_df, pd.DataFrame(true_labels, columns=['true_lineup_pos'])], axis=1)

final_df.to_parquet("data/model_preds.parquet", index=False)
final_df[:50].to_csv("data/model_preds_preview.csv", index=False)

print("Model predictions saved to model_preds.parquet")

all_preds = []
all_true = []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        logits = model(X_batch)
        preds = ordinal_to_class(logits)

        all_preds.extend(preds.numpy())
        all_true.extend((y_batch.sum(dim=1) + 1).numpy())

all_preds = np.array(all_preds)
all_true = np.array(all_true)

model_output = f"""
Final Model Evaluation:
MSE: {mean_squared_error(all_true, all_preds):.4f}
MAE: {mean_absolute_error(all_true, all_preds):.4f}
R^2: {r2_score(all_true, all_preds):.4f}
Within ±1 Accuracy: {np.mean(np.abs(all_preds - all_true) <= 1):.4f}
"""
print(model_output)
with open("data/model_evaluation.txt", "w") as f:
    f.write(model_output)

#Predict team lineup
def create_teams_df(df, feature_cols):
    def make_lineup(group):
        if len(group) != 9:
            return None
        return {row['player_id']: row[feature_cols].tolist() for _, row in group.iterrows()}
    
    teams_df = df.groupby(['team', 'game_date']).apply(make_lineup, include_groups=False).dropna().reset_index(name='lineup')
    return teams_df

teams_df = create_teams_df(df, feature_cols)

def get_position_probs(logits):
    # Logits already represent (base - thresholds) from the forward pass.
    # The Dataset encoding uses ordinal[:y-1] = 1, meaning logits[k] targets P(pos > k+1)
    s = torch.sigmoid(logits)
    probs = torch.zeros(9)
    
    # P(pos=1) = 1 - P(pos > 1)
    probs[0] = 1 - s[0]
    for i in range(1, 8):
        # P(pos=i+1) = P(pos > i) - P(pos > i+1)
        probs[i] = s[i-1] - s[i]
    # P(pos=9) = P(pos > 8)
    probs[8] = s[7]
    return probs.detach().numpy()

def predict_lineup(player_ids, player_features):
    if len(player_ids) != 9 or player_features.shape[0] != 9:
        return None
    
    player_features = scaler.transform(player_features)
    X_tensor = torch.tensor(player_features, dtype=torch.float32)
    
    with torch.no_grad():
        logits = model(X_tensor)
        prob_matrix = np.array([get_position_probs(logits[i]) for i in range(9)])
    
    cost_matrix = -prob_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignment = {row: col + 1 for row, col in zip(row_ind, col_ind)}
    
    predicted_lineup = [None] * 9
    for idx, pos in assignment.items():
        predicted_lineup[pos - 1] = player_ids[idx]
    return predicted_lineup

predicted_lineups = []
for _, row in tqdm(teams_df.iterrows(), desc="Predicting Lineups", total=len(teams_df)):
    lineup_dict = row['lineup']
    player_ids = list(lineup_dict.keys())
    features = np.array([lineup_dict[pid] for pid in player_ids])
    pred_lineup = predict_lineup(player_ids, features)
    predicted_lineups.append(pred_lineup)

teams_df['predicted_lineup'] = predicted_lineups
teams_df['lineup'] = teams_df['lineup'].apply(lambda d: list(d.keys()))
teams_df.to_parquet("data/predicted_lineups.parquet", index=False)
teams_df[:100].to_csv("data/predicted_lineups_preview.csv", index=False)