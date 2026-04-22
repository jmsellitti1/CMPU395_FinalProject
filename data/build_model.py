import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load Data
df = pd.read_parquet('data/features.parquet')

feature_cols = [
    'AVG', 'SLG', 'OBP', 'HR', 'xwOBA',
    'K%', 'BB%', 'K/BB', 'Contact%', 'SweetSpot%', 'HardHit%'
]

X = df[feature_cols].values
y = df['lineup_pos'].astype(int).values
df['game_id'] = df['team'] + "_" + df['game_date'].astype(str)

# Create groups to ensure all 9 players from a single game stay together in the same split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups=df['game_id']))

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]
val_df = df.iloc[val_idx].copy()

# feature normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_all = scaler.transform(X)

class PlayerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y - 1, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(PlayerDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(PlayerDataset(X_val, y_val), batch_size=32)

# Player Model
class PlayerModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.embedding = nn.Linear(64, 32)
        self.out = nn.Linear(32, 9)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        emb = F.relu(self.embedding(x))
        logits = self.out(emb)
        return logits, emb

player_model = PlayerModel(X.shape[1])
optimizer = torch.optim.Adam(player_model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

num_epochs = 100
best_val_loss = float('inf')
epoch_losses = {}
best_state = None

for epoch in tqdm(range(num_epochs), desc="Training Player Model"):
    player_model.train()
    train_loss = 0

    for X_batch, y_batch in train_loader:
        logits, _ = player_model(X_batch)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    player_model.eval()
    val_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits, _ = player_model(X_batch)
            val_loss += criterion(logits, y_batch).item()

    val_loss /= len(val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = player_model.state_dict()
    
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

player_model.load_state_dict(best_state)
all_preds, all_true, all_probs = [], [], []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        logits, _ = player_model(X_batch)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.numpy())
        all_true.extend(y_batch.numpy())
        all_probs.extend(probs.numpy())

all_preds = np.array(all_preds)
all_true = np.array(all_true)
all_probs = np.array(all_probs)

player_top1 = np.mean(all_preds == all_true)
player_top3 = np.mean([
    all_true[i] in np.argsort(all_probs[i])[-3:]
    for i in range(len(all_true))
])
player_mae = np.mean(np.abs(all_preds - all_true))

def create_teams_df(df):
    teams = df.groupby(['team', 'game_date']).filter(lambda g: len(g) == 9).copy()
    teams['game_id'] = teams['team'] + "_" + teams['game_date'].astype(str)
    return teams

teams_df = create_teams_df(df)
train_game_ids = set(df.iloc[train_idx]['game_id'])
val_game_ids = set(df.iloc[val_idx]['game_id'])
train_teams_df = teams_df[teams_df['game_id'].isin(train_game_ids)]
val_teams_df = teams_df[teams_df['game_id'].isin(val_game_ids)]

class TeamDataset(Dataset):
    def __init__(self, teams_df):
        self.samples = []

        for _, group in teams_df.groupby(['team', 'game_date']):
            if len(group) != 9:
                continue

            features = scaler.transform(group[feature_cols].values)
            labels = group['lineup_pos'].values - 1

            self.samples.append((features, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

train_team_dataset = TeamDataset(train_teams_df)
val_team_dataset = TeamDataset(val_teams_df)

train_team_loader = DataLoader(train_team_dataset, batch_size=16, shuffle=True)
val_team_loader = DataLoader(val_team_dataset, batch_size=16)

# Team Model
class TeamModel(nn.Module):
    def __init__(self, player_model):
        super().__init__()
        self.player_model = player_model
        self.scorer = nn.Linear(32, 9)

    def forward(self, x):
        B, N, D = x.shape
        x = x.view(B * N, D)

        with torch.no_grad():
            _, emb = self.player_model(x)

        scores = self.scorer(emb)
        return scores.view(B, N, 9)

team_model = TeamModel(player_model)
optimizer = torch.optim.Adam(team_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in tqdm(range(num_epochs), desc="Training Team Model"):
    team_model.train()
    total_loss = 0

    for X_batch, y_batch in train_team_loader:
        logits = team_model(X_batch)

        loss = 0
        for i in range(9):
            loss += criterion(logits[:, i, :], y_batch[:, i])
        loss /= 9

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

# Evaluate Team Model
exact_match = 0
displacements = []
pairwise_scores = []

for X_batch, y_batch in val_team_loader:
    with torch.no_grad():
        logits = team_model(X_batch)
        probs = torch.softmax(logits, dim=2).numpy()

    for b in range(len(X_batch)):
        cost = -probs[b]
        row_ind, col_ind = linear_sum_assignment(cost)

        pred = np.zeros(9, dtype=int)
        for r, c in zip(row_ind, col_ind):
            pred[r] = c

        true = y_batch[b].numpy()

        if np.array_equal(pred, true):
            exact_match += 1

        displacements.append(np.mean(np.abs(pred - true)))

        correct, total = 0, 0
        for i in range(9):
            for j in range(i+1, 9):
                total += 1
                if (pred[i] < pred[j]) == (true[i] < true[j]):
                    correct += 1
        pairwise_scores.append(correct / total)

team_exact = exact_match / len(val_team_dataset)
team_disp = np.mean(displacements)
team_pairwise = np.mean(pairwise_scores)

output = f"""PLAYER MODEL:
Exact Accuracy: {player_top1:.4f}
Top-3 Accuracy (Actual position in top 3 predicted positions): {player_top3:.4f}
MAE: {player_mae:.4f}

TEAM MODEL:
Exact Match %: {team_exact:.4f}
Avg Displacement/Position Error: {team_disp:.4f}
Pairwise Accuracy (Predicting which player in each pair of 2 bats before the other): {team_pairwise:.4f}"""
print(output)
with open("data/model_evaluation.txt", "w") as f:
    f.write(output)