from pybaseball import statcast, statcast_batter
import pandas as pd

data = statcast('2025-05-01')
data.to_csv('statcast_data.csv', index=False)
player_id = data['batter'].unique()[0]
batting_stats = statcast_batter('2025-01-01', '2025-12-31', player_id)
batting_stats.to_csv(f'{player_id}_batting_stats.csv', index=False)