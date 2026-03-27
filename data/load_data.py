from pybaseball import statcast, statcast_batter, retrosheet
import pandas as pd
import csv
from io import StringIO
import datetime

# LOAD STATCAST DATA FOR EVERY PITCH ON 5/1/2025
# data = statcast('2025-05-01')
# data.to_csv('statcast_data.csv', index=False)

# LOAD 10 PLAYERS' BATTING STATS FOR 2025 SEASON
# for i in range(10):
#     player_id = data['batter'].unique()[i]
#     batting_stats = statcast_batter('2025-01-01', '2025-12-31', player_id)
#     batting_stats.to_csv(f'playerStats/{player_id}_batting_stats.csv', index=False)

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