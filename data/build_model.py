import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

df = pd.read_parquet('data/features.parquet')

# Features
feature_cols = [
    'AVG', 'SLG', 'OBP', 'HR', 'xwOBA',
    'K%', 'BB%', 'Contact%', 'SweetSpot%', 'HardHit%'
]

X = df[feature_cols].values
# convert lineup position to 0-8 rather then 1-9 so easier to handle
y = df['lineup_pos'].astype(int).values - 1

# train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# feature normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X = scaler.transform(X)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_tensor = torch.tensor(X, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

class LineupModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 9)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x) # logits

model = LineupModel(X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 2000
patience = 20
min_delta = 1e-3 # require meaningful improvement
best_val_loss = float('inf')
counter = 0
best_model_state = None

epoch_losses = {}
for epoch in tqdm(range(num_epochs), desc="Training Model"):
    # Training
    logits = model(X_train_tensor)
    loss = criterion(logits, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Validation
    with torch.no_grad():
        val_logits = model(X_val_tensor)
        val_loss = criterion(val_logits, y_val_tensor).item()
    
    # Early stopping check
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        counter = 0
        best_model_state = model.state_dict().copy()
    else:
        counter += 1
    if counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
    
    if epoch % 20 == 19:
        epoch_losses[epoch+1] = {
            'train_loss': loss.item(),
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

with torch.no_grad():
    probs = F.softmax(model(X_tensor), dim=1).numpy()
    
prob_cols = [f'pos_{i+1}' for i in range(9)]
probs_df = pd.DataFrame(probs, columns=prob_cols)
    
# Attach metadata
metadata = df[['team', 'player_id', 'game_date']].reset_index(drop=True)
final_df = pd.concat([metadata, probs_df], axis=1)
    
# Overwrite file every run
final_df.to_parquet("data/model_probs.parquet", index=False)
final_df[:50].to_csv("data/model_probs_preview.csv", index=False)
print("Model probabilities saved to model_probs.parquet")

# Final Model Evaluation
with torch.no_grad():
    final_logits = model(X_val_tensor)
    final_preds = torch.argmax(final_logits, dim=1).numpy()
print("Final Model Evaluation:")
print(f"MSE: {mean_squared_error(y_val, final_preds):.4f}")
print(f"MAE: {mean_absolute_error(y_val, final_preds):.4f}")
print(f"R^2: {r2_score(y_val, final_preds):.4f}")