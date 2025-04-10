import pandas as pd

df = pd.read_csv('Raw Data/Tournament Matchups.csv')

matchups = df.iloc[::2].reset_index(drop=True)

matchups["TEAM2"] = df.iloc[1::2]["TEAM"].values
matchups["SEED2"] = df.iloc[1::2]["SEED"].values
matchups["SCORE2"] = df.iloc[1::2]["SCORE"].values

matchups.rename(
    columns={"TEAM": "TEAM1", "SEED": "SEED1", "SCORE": "SCORE1"},
    inplace=True
)

matchups['RESULT'] = (matchups['SCORE1'] > matchups['SCORE2']).astype(int)
matchups.drop(columns=['BY YEAR NO', 'BY ROUND NO', 'ROUND', 'TEAM NO', 'SCORE1', 'SCORE2'], inplace=True)

matchups_2025 = matchups[matchups['YEAR'] == 2025]

matchups_2025.to_csv('Transformed Data/2025 Matchups.csv', index=False)

matchups = matchups[(matchups['YEAR'] <= 2023) & (matchups['YEAR'] >= 2013)].reset_index(drop=True)

historical_stats = pd.read_csv('Raw Data/Historical Stats.csv')

historical_stats.drop(columns=['POSTSEASON', 'SEED'], inplace=True)

stats1 = historical_stats.rename(columns=lambda x: x + "1" if x != 'YEAR' else x)
stats2 = historical_stats.rename(columns=lambda x: x + "2" if x != 'YEAR' else x)

combined_data = matchups.merge(stats1, how='left', on=['TEAM1', 'YEAR']).merge(stats2, how='left', on=['TEAM2', 'YEAR'])
training_data = combined_data.dropna().reset_index(drop=True)
training_data.drop(columns=['YEAR', 'TEAM1', 'TEAM2', 'CONF1', 'CONF2', 'CURRENT ROUND'], inplace=True)

training_data.to_csv('Transformed Data/Training Data.csv', index=False)