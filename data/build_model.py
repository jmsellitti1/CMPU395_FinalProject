import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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

num_epochs = 250
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

    if epoch % 10 == 9:
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

print("Final Model Evaluation:")
print(f"MSE: {mean_squared_error(all_true, all_preds):.4f}")
print(f"MAE: {mean_absolute_error(all_true, all_preds):.4f}")
print(f"R^2: {r2_score(all_true, all_preds):.4f}")

within_1 = np.mean(np.abs(all_preds - all_true) <= 1)
print(f"Within ±1 Accuracy: {within_1:.4f}")