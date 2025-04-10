import json
import pandas as pd

with open('game_slot_probabilities.json', 'r') as f:
    game_probabilities = json.load(f)

rows = []
for game, probs in game_probabilities.items():
    most_common_team, win_percentage = max(probs.items(), key=lambda x: x[1])
    rows.append({'Game': game, 'Most Common Winner': most_common_team, 'Win Percentage (%)': win_percentage})

df = pd.DataFrame(rows)

df['Game Number'] = df['Game'].str.extract('(\d+)').astype(int)
df.sort_values('Game Number', inplace=True)
df.drop('Game Number', axis=1, inplace=True)

print(df.to_string(index=False))