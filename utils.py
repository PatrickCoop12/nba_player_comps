from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats
import numpy as np
import pandas as pd

all_players = players.get_players()

# %%
inactive_players_ids = {}
active_players_ids = {}

for player in all_players:
    if player.get('is_active') == False:

        inactive_players_ids[f'{player.get("full_name").lower()}'] = player.get("id")

    else:

        active_players_ids[f'{player.get("full_name").lower()}'] = player.get("id")

df2 = pd.read_csv('Player Totals.csv')
df2.head()
#%%
career_stats = df2.groupby('player', as_index=False).mean(numeric_only=True).drop(columns=['season', 'age', 'x2p', 'x2pa','x2p_percent', 'e_fg_percent', 'trp_dbl'])

career_stats['status'] = career_stats['player'].apply(lambda x: 'inactive' if x.lower() in list(inactive_players_ids.keys()) else 'active')
#%%
retired = career_stats.loc[career_stats['status'] == 'inactive']
actives = career_stats.loc[career_stats['status'] == 'active']

def pull_player_ids():
    n_players = players.get_active_players()

    n_players_ids = {}

    for player in n_players:
        n_players_ids[f'{player.get("full_name").lower()}'] = player.get("id")

    return n_players_ids

N_PLAYERS = pull_player_ids()


def pull_player_career_stats(player_name):
    id = active_players_ids[player_name.lower()]
    career = playercareerstats.PlayerCareerStats(player_id=str(id))

    c = career.season_totals_regular_season.get_data_frame()

    c = c.groupby('PLAYER_ID').mean(numeric_only=True).iloc[:, 1:].reset_index().iloc[:, 2:]

    c.insert(0, 'name', player_name)
    # c.insert(0,'id', player_id)

    return c

def get_player_comps(player_name):
    player_career_stats = pull_player_career_stats(player_name)

    player_career_stats_clean = player_career_stats.drop(columns=['name'])

    inactives = retired.copy()

    inactives2 = inactives.drop(columns=['player', 'status'])

    matrix = inactives2.to_numpy()
    print(matrix.shape)
    vec = player_career_stats_clean.to_numpy().flatten()
    print(vec.shape)

    dot = np.dot(matrix, vec)
    norms = np.linalg.norm(matrix, axis=1)
    vec_norm = np.linalg.norm(vec)

    similarity = dot / (norms * vec_norm)

    inactives['similarity'] = similarity

    inactives = inactives.sort_values(by='similarity', ascending=False)
    return player_career_stats, inactives

