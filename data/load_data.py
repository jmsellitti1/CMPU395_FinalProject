from pybaseball import statcast
import pandas as pd
import csv
from io import StringIO
import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# LOAD STATCAST DATA FOR 2025 SEASON
season_2025 = statcast('2025-01-01', '2025-12-31')
# Only get regular season events where something happened (not just a ball or strike)
season_2025_events = season_2025[season_2025['events'].notna()]
season_2025_events = season_2025_events[season_2025_events['game_type'] == 'R']
# Filter by batters who had more than 250 plate appearances (~1.5 per game)
data = season_2025_events[season_2025_events['batter'].isin(
    season_2025_events['batter'].value_counts()[season_2025_events['batter'].value_counts() > 250].index)]
data = data.sort_values(by=['batter', 'game_date'])
data.to_parquet('data/season_2025.parquet', index=False)
data.head(50).to_csv('data/season_2025_preview.csv', index=False) #For testing + visualization purposes
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
        away_lineup.append(game[106 + i*3])
        home_lineup.append(game[133 + i*3])
    date = datetime.datetime.strptime(game[0], '%Y%m%d').strftime('%m/%d/%Y')
    rows.append({
        'date': date,
        'away_lineup': away_lineup,
        'home_lineup': home_lineup
    })

lineups = pd.DataFrame(rows)
lineups.to_csv('data/lineups.csv', index=False)
print("Lineups saved to lineups.csv")