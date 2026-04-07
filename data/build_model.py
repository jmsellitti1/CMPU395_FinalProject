# multi class classification model

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_parquet('features.parquet')


# Features
feature_cols = [
    'AVG', 'SLG', 'OBP', 'HR', 'xwOBA',
    'K%', 'BB%', 'Contact%', 'SweetSpot%', 'HardHit%'
]

X = df[feature_cols].values

# convert lineup position to 0-8 rather then 1-9 so easier to handle
y = df['lineup_pos'].astype(int).values - 1



#feature normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)


import torch
import torch.nn as nn
import torch.nn.functional as F

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

class LineupModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 9)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)  # logits (NOT softmax here)

model = LineupModel(X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


for epoch in range(50):
    logits = model(X_tensor)
    loss = criterion(logits, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")



with torch.no_grad():
    probs = F.softmax(model(X_tensor), dim=1).numpy()
    
   
prob_cols = [f'pos_{i+1}' for i in range(9)]
probs_df = pd.DataFrame(probs, columns=prob_cols)
    
# Attach metadata
metadata = df[['team', 'player_id', 'game_date']].reset_index(drop=True)
    
    
final_df = pd.concat([metadata, probs_df], axis=1)
    
# Overwrite file every run
final_df.to_csv("model_probs.csv", index=False)  # this clears + rewrites automatically


