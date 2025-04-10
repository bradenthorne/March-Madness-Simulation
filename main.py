import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import json
from model import Model
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

model = Model()

training_data = pd.read_csv('Transformed Data/Training Data.csv')
model.fit(training_data)

stats = pd.read_csv('Transformed Data/2025 Stats.csv')
model.load_stats(stats)

initial_matchups = pd.read_csv('Transformed Data/Corrected 2025 Matchups.csv')

# print(model.predict(initial_matchups, 'winner'))

'''
def simulate_tournament():
    local_results = defaultdict(Counter)
    current_round = initial_matchups.copy()
    game_number = 1

    while len(current_round) > 1:
        probs_df = model.predict(current_round, 'probability')
        team1_probs = probs_df['TEAM1 WIN PROBABILITY'].values
        team1_names = probs_df['TEAM1'].values
        team2_names = probs_df['TEAM2'].values

        random_draws = np.random.rand(len(probs_df))
        team1_wins = random_draws < team1_probs
        winners = np.where(team1_wins, team1_names, team2_names)

        for winner in winners:
            local_results[f'Game {game_number}'][winner] += 1
            game_number += 1

        current_round = pd.DataFrame({
            'TEAM1': winners[::2],
            'TEAM2': winners[1::2]
        })

    if len(current_round) == 1:
        probs_df = model.predict(current_round, 'probability')
        team1_prob = probs_df['TEAM1 WIN PROBABILITY'].iloc[0]
        team1_name = probs_df['TEAM1'].iloc[0]
        team2_name = probs_df['TEAM2'].iloc[0]
        random_draw = np.random.rand()
        champion = team1_name if random_draw < team1_prob else team2_name
        local_results[f'Game {game_number}'][champion] += 1

    return local_results

if __name__ == '__main__':
    num_simulations = 10000
    max_workers = 8
    final_results = defaultdict(Counter)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(simulate_tournament) for _ in range(num_simulations)]
        for future in tqdm(as_completed(futures), total=num_simulations, desc="Simulating tournaments"):
            local_result = future.result()
            for game, counts in local_result.items():
                final_results[game].update(counts)

    game_probabilities = {}
    for game, counts in final_results.items():
        total = sum(counts.values())
        game_probabilities[game] = {team: round(count / total * 100, 2) for team, count in counts.items()}

    with open('game_slot_probabilities.json', 'w') as f:
        json.dump(game_probabilities, f, indent=2)

    print("Simulation complete. Results saved to 'game_slot_probabilities.json'")'
'''