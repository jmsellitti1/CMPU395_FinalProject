import ast
import pandas as pd
import numpy as np
from tqdm import tqdm

events = pd.read_parquet('data/season_2025.parquet')
lineups = pd.read_csv('data/lineups.csv')
features_df = []

def get_lineup_pos(batter_id, lineups, side):
    for _, row in lineups.iterrows():
        lineup = row[side]
        if isinstance(lineup, str):
            lineup = ast.literal_eval(lineup)
        if batter_id in lineup:
            return lineup.index(batter_id) + 1
    return np.nan

for player_id, group in tqdm(events.groupby('batter'), desc="Calculating features"):
    group = group.sort_values('game_date').copy()

    # Flag variables
    group['is_hit'] = group['events'].isin(['single', 'double', 'triple', 'home_run']).astype(int)
    group['is_ab'] = (~group['events'].isin(['walk', 'hit_by_pitch', 'sac_bunt', 'sac_fly'])).astype(int)
    group['is_bb'] = (group['events'] == 'walk').astype(int)
    group['is_k'] = (group['events'] == 'strikeout').astype(int)
    group['is_hr'] = (group['events'] == 'home_run').astype(int)
    group['is_hbp'] = (group['events'] == 'hit_by_pitch').astype(int)
    group['pa'] = group['is_ab'] + group['is_bb'] + group['is_hbp']
    group['is_bip'] = group['events'].isin(['single','double','triple','home_run',
                                            'field_out','force_out','grounded_into_double_play']).astype(int)
    group['tb'] = (
        (group['events'] == 'single').astype(int)
        + 2*(group['events'] == 'double').astype(int)
        + 3*(group['events'] == 'triple').astype(int)
        + 4*(group['events'] == 'home_run').astype(int)
    )

    # Cumulative stats up to that point
    c_hits = group['is_hit'].cumsum().shift(1).fillna(0)
    c_ab = group['is_ab'].cumsum().shift(1).fillna(0)
    c_tb = group['tb'].cumsum().shift(1).fillna(0)
    c_bb = group['is_bb'].cumsum().shift(1).fillna(0)
    c_k = group['is_k'].cumsum().shift(1).fillna(0)
    c_hbp = group['is_hbp'].cumsum().shift(1).fillna(0)
    c_pa = group['pa'].cumsum().shift(1).fillna(0)
    c_hr = group['is_hr'].cumsum().shift(1).fillna(0)
    c_bip = group['is_bip'].cumsum().shift(1).fillna(0)

    # Safe division helper
    def safe_div(num, denom):
        return np.where(denom > 0, num / denom, 0)

    group['AVG'] = safe_div(c_hits, c_ab)
    group['SLG'] = safe_div(c_tb, c_ab)
    group['OBP'] = safe_div(c_hits + c_bb + c_hbp, c_pa)
    group['HR'] = c_hr
    group['K%'] = safe_div(c_k, c_pa)
    group['BB%'] = safe_div(c_bb, c_pa)
    group['Contact%'] = safe_div((c_pa - c_k), c_pa)
    group['sweet_spot'] = (group['launch_angle'].between(8, 32).fillna(False) & group['is_bip'].astype(bool)).astype(int)
    c_sweet = group['sweet_spot'].cumsum().shift(1).fillna(0)
    group['SweetSpot%'] = safe_div(c_sweet, c_bip)
    group['hard_hit'] = ((group['launch_speed'] >= 95).fillna(False) & group['is_bip'].astype(bool)).astype(int)
    c_hard = group['hard_hit'].cumsum().shift(1).fillna(0)
    group['HardHit%'] = safe_div(c_hard, c_bip)
    group['xwoba_est'] = (
        0.7*group['is_bb'] +
        0.9*(group['events'] == 'single').astype(int) +
        1.3*(group['events'] == 'double').astype(int) +
        1.6*(group['events'] == 'triple').astype(int) +
        2.0*(group['events'] == 'home_run').astype(int)
    )
    c_xwoba = group['xwoba_est'].cumsum().shift(1).fillna(0)
    group['xwOBA'] = safe_div(c_xwoba, c_pa)
    
    group['team'] = group['away_team'].where(group['inning_topbot'] == 'Top', group['home_team'])

    # Fetch lineup position
    group['lineup_pos'] = np.nan
    # Iterate through all games for that player
    for gdate in group['game_date'].unique():
        mask = group['game_date'] == gdate
        game_date_str = pd.to_datetime(gdate).strftime('%Y-%m-%d')
        batter_id = group.loc[mask, 'batter'].iloc[0]
        side = 'away_lineup' if group.loc[mask, 'inning_topbot'].iloc[0] == 'Top' else 'home_lineup'
        lineup_pos = get_lineup_pos(batter_id, lineups[lineups['date'] == game_date_str], side)
        group.loc[mask, 'lineup_pos'] = lineup_pos
    
    # Final features
    group = group[['game_date', 'batter', 'team', 'AVG', 'SLG', 'OBP', 'HR', 'xwOBA', 'K%', 'BB%', 'Contact%', 'SweetSpot%', 'HardHit%', 'lineup_pos']]
    # Drop first 10 games of season to allow for normalization of features
    group = group.iloc[10:]
    # Remove any rows where lineup_pos is not found - pinch hitters
    group = group[group['lineup_pos'].notna()]
    group = group.groupby('game_date').first().reset_index()
    features_df.append(group)

features_df = pd.concat(features_df)
features_df.rename(columns={'batter': 'player_id'}, inplace=True)
features_df = features_df.sort_values(['game_date', 'team'])
features_df.to_parquet('data/features.parquet', index=False)
features_df.head(50).to_csv('data/features_preview.csv', index=False)
print("Features saved to features.parquet")