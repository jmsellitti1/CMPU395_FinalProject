import pandas as pd
from pybaseball import statcast, playerid_reverse_lookup
import csv
from io import StringIO
import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# LOAD STATCAST DATA FOR 2025 SEASON
season_2025 = statcast('2025-03-01', '2025-11-01')
# Only get regular season events where something happened (not just a ball or strike)
season_2025_events = season_2025[season_2025['events'].notna()]
season_2025_events = season_2025_events[season_2025_events['game_type'] == 'R']
# Filter by batters who had more than 100 total plate appearances
data = season_2025_events[season_2025_events['batter'].isin(
    season_2025_events['batter'].value_counts()[season_2025_events['batter'].value_counts() > 100].index)].copy()

# Convert MLB IDs to Retrosheet IDs for easier lineup merging
unique_players = data['batter'].dropna().unique().tolist()
player_map = playerid_reverse_lookup(unique_players)
player_map = player_map[['key_mlbam', 'key_retro']]
mlb_to_retro = dict(zip(player_map['key_mlbam'], player_map['key_retro']))
data['batter'] = data['batter'].map(mlb_to_retro)
data = data[data['batter'].notna()]

# Sort by date, team, and inning (assending)
data = data.sort_values(by=['game_date', 'home_team', 'inning'])
data.to_parquet('data/season_2025.parquet', index=False)
data[:100].to_csv('data/season_2025_preview.csv', index=False)
print("Statcast data saved to season_2025.parquet")

# LOAD GAME DATA TO PARSE BATTING ORDERS
with open('data/gl2025.txt', 'r') as file:
    lines = file.readlines()
parsed = []
for line in lines:
    parsed.extend(list(csv.reader(StringIO(line))))

rows = []
for game in parsed:
    away_lineup = []
    home_lineup = []
    for i in range(9):
        away_lineup.append(game[105 + i*3])
        home_lineup.append(game[132 + i*3])
    date = datetime.datetime.strptime(game[0], '%Y%m%d').strftime('%Y-%m-%d')
    rows.append({
        'date': date,
        'away_lineup': away_lineup,
        'home_lineup': home_lineup
    })

lineups = pd.DataFrame(rows)
lineups.to_csv('data/lineups.csv', index=False)
print("Lineups saved to lineups.csv")