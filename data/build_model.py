import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
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
val_df = df.iloc[val_idx].copy().reset_index(drop=True)

# feature normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

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

all_preds_1to9 = all_preds + 1
all_true_1to9 = all_true + 1

player_preds_df = pd.DataFrame({
    "team": val_df['team'].values,
    "player_id": val_df['player_id'].values,
    "game_date": val_df['game_date'].astype(str).values,
    "predicted_lineup_pos": all_preds_1to9,
    "true_lineup_pos": all_true_1to9
})
player_preds_df.to_parquet("data/model_preds.parquet", index=False)
player_preds_df.head(100).to_csv("data/model_preds_preview.csv", index=False)
print("Player predictions saved to model_preds.parquet")

teams_df = df.groupby(['team', 'game_date']).filter(lambda g: len(g) == 9).copy()
teams_df['game_id'] = teams_df['team'] + "_" + teams_df['game_date'].astype(str)

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

            group = group.sample(frac=1, random_state=42).reset_index(drop=True)
            features = scaler.transform(group[feature_cols].values)
            labels = group['lineup_pos'].values - 1
            player_ids = group['player_id'].values
            team = group['team'].iloc[0]
            game_date = group['game_date'].iloc[0]

            self.samples.append((features, labels, player_ids, team, game_date))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y, pids, team, date = self.samples[idx]
        return (torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long),
            pids, team, date)

train_team_dataset = TeamDataset(train_teams_df)
val_team_dataset = TeamDataset(val_teams_df)

def team_collate_fn(batch):
    X, y, pids, teams, dates = zip(*batch)
    X = torch.stack(X)
    y = torch.stack(y)
    pids = list(pids)
    teams = list(teams)
    dates = list(dates)
    return X, y, pids, teams, dates

train_team_loader = DataLoader(train_team_dataset, batch_size=16,
                               shuffle=True, collate_fn=team_collate_fn)

val_team_loader = DataLoader(val_team_dataset, batch_size=16,
                             shuffle=False, collate_fn=team_collate_fn)

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

    for X_batch, y_batch, _, _, _ in train_team_loader:
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
lineup_results = []

for X_batch, y_batch, pids_batch, team_batch, date_batch in val_team_loader:
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
        player_ids = list(pids_batch[b])
        true_lineup = [None] * 9
        for i, pos in enumerate(true):
            true_lineup[pos] = player_ids[i]
        pred_lineup = [None] * 9
        for player_idx, pos in enumerate(pred):
            pred_lineup[pos] = player_ids[player_idx]

        # Save result
        lineup_results.append({
            "team": team_batch[b],
            "game_date": str(date_batch[b]),
            "lineup": true_lineup,
            "predicted_lineup": pred_lineup
        })

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

def evaluate_model(X, y):
    with torch.no_grad():
        logits, _ = player_model(torch.tensor(X, dtype=torch.float32))
        preds = torch.argmax(logits, dim=1).numpy()
    return mean_absolute_error(y, preds)
base_mae = evaluate_model(X_val, y_val)
importances = []
for i, col in enumerate(feature_cols):
    X_perm = X_val.copy()
    np.random.shuffle(X_perm[:, i])

    perm_mae = evaluate_model(X_perm, y_val)
    importance = perm_mae - base_mae
    importances.append((col, importance))
importances.sort(key=lambda x: x[1], reverse=True)

lineup_df = pd.DataFrame(lineup_results)
lineup_df.to_parquet("data/predicted_lineups.parquet", index=False)
lineup_df.head(100).to_csv("data/predicted_lineups_preview.csv", index=False)
print("Predicted lineups saved to predicted_lineups.parquet")

output = f"""PLAYER MODEL:
Exact Accuracy: {player_top1:.4f}
Top-3 Accuracy (Actual position in top 3 predicted positions): {player_top3:.4f}
MAE: {player_mae:.4f}

TEAM MODEL:
Exact Match %: {team_exact:.4f}
Avg Displacement/Position Error: {team_disp:.4f}
Pairwise Accuracy (Predicting which player in each pair of 2 bats before the other): {team_pairwise:.4f}\n
Feature Importances:\n"""

for feat, imp in importances:
    output += f"{feat}: {imp:.4f}\n"
    
scores = {i: 0 for i in range(1, 10)}
for _, row in lineup_df.iterrows():
    pred_lineup = row['predicted_lineup']
    for pos in range(1, 10):
        if row['lineup'][pos-1] == pred_lineup[pos-1]:
            scores[pos] += 1
            
total_correct = sum(scores.values())
total_slots = 9 * len(lineup_df)
output += f"\nOverall Player Placement Accuracy: {total_correct}/{total_slots} ({(total_correct/total_slots)*100:.2f}%)\n"

output += "\nLineup Position Accuracy:\n"
for pos in range(1, 10):
    output += f"Position {pos}: {scores[pos] / len(lineup_df)*100:.2f}%\n"
    
conf_matrix = pd.DataFrame(0, index=range(1, 10), columns=range(1, 10))
for _, row in lineup_df.iterrows():
    pred_lineup = row['predicted_lineup']
    true_lineup = row['lineup']
    for true_pos in range(1, 10):
        true_player = true_lineup[true_pos-1]
        for pred_pos in range(1, 10):
            if pred_lineup[pred_pos-1] == true_player:
                conf_matrix.loc[true_pos, pred_pos] += 1
output += "\nConfusion Matrix: True Position (Rows) vs Predicted Position (Columns):\n"
output += str(conf_matrix)

print(output)
with open("data/model_evaluation.txt", "w") as f:
    f.write(output)