import pandas as pd

output = ""

teams_df = pd.read_parquet("data/predicted_lineups.parquet")
scores = {i: 0 for i in range(1, 10)}
for _, row in teams_df.iterrows():
    pred_lineup = row['predicted_lineup']
    for pos in range(1, 10):
        if row['lineup'][pos-1] == pred_lineup[pos-1]:
            scores[pos] += 1
output += "\nLineup Position Accuracy:\n"
for pos in range(1, 10):
    output += f"Position {pos}: {scores[pos] / len(teams_df)*100:.2f}%\n"
    
conf_matrix = pd.DataFrame(0, index=range(1, 10), columns=range(1, 10))
for _, row in teams_df.iterrows():
    pred_lineup = row['predicted_lineup']
    true_lineup = row['lineup']
    for true_pos in range(1, 10):
        true_player = true_lineup[true_pos-1]
        for pred_pos in range(1, 10):
            if pred_lineup[pred_pos-1] == true_player:
                conf_matrix.loc[true_pos, pred_pos] += 1
output += "\nConfusion Matrix: True Position (Rows) vs Predicted Position (Columns):\n"
output += str(conf_matrix)
with open("data/model_evaluation.txt", "a") as f:
    f.write(output + "\n")
print(output)