import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import glm
from scipy.stats import poisson
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from sklearn.preprocessing import StandardScaler

def update_elo(r_home, r_away, goal_diff, k):
    # https://en.wikipedia.org/wiki/Elo_rating_system
    if goal_diff > 0:
        score_home, score_away = 1, 0
    elif goal_diff < 0:
        score_home, score_away = 0, 1
    else:
        score_home, score_away = 0.5, 0.5

    # expected score formula from wikipedia
    expected_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
    expected_away = 1 - expected_home

    # update the ratings
    r_home = r_home + k * (score_home - expected_home)
    r_away = r_away + k * (score_away - expected_away)

    return r_home, r_away #, expected_home, expected_away

def get_elo(df, k=20):
    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel("K"))
    
    elo_ratings = {team: 1500 for team in teams} # Init
    elo_history = []

    for i, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        r_home, r_away = elo_ratings[home], elo_ratings[away]

        fthg, ftag = row["FTHG"], row["FTAG"]
        goal_diff = fthg - ftag
        elo_diff = r_home - r_away

        r_home_new, r_away_new = update_elo(r_home, r_away, goal_diff,k=k)
        elo_ratings[home], elo_ratings[away] = r_home_new, r_away_new

        elo_history.append({
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": fthg,
            "FTAG": ftag,
            "EloDiff": elo_diff, # elo difference before this game
        })
    
    return pd.DataFrame(elo_history), elo_ratings

def _evaluate_k(df, k, start_index, max_goals, goal_target):
    """
    Compute log-likelihood for a single k.
    Returns (total_LL, k, elo_history) so the caller can keep the best one.
    Everything here runs in a separate process.
    """
    elo_history, elo_ratings = get_elo(df, k=k)
    elo_history = elo_history[start_index:].reset_index(drop=True)

    scaler = StandardScaler()
    elo_history["EloDiff"] = scaler.fit_transform(elo_history["EloDiff"].values.reshape(-1, 1))

    home_model = glm("FTHG ~ EloDiff", data=elo_history,
                    family=sm.families.Poisson()).fit()
    away_model = glm("FTAG ~ EloDiff", data=elo_history,
                    family=sm.families.Poisson()).fit()

    total_LL = 0.0
    for _, row in elo_history.iterrows():
        elo_diff = row["EloDiff"]
        home_pred = home_model.predict({"EloDiff": elo_diff})[0]
        away_pred = away_model.predict({"EloDiff": elo_diff})[0]

        # compute under/over probability
        prob_under = sum(
            poisson.pmf(i, home_pred) * poisson.pmf(j, away_pred)
            for i in range(max_goals + 1)
            for j in range(max_goals + 1)
            if i + j <= goal_target
        )
        prob_over = 1 - prob_under

        actual_goals = row["FTHG"] + row["FTAG"]
        p = prob_over if actual_goals > goal_target else prob_under
        total_LL += np.log(max(p, 1e-12)) # avoid log(0)

    return total_LL, k, elo_history, elo_ratings
    
def fit_models(df, goal_target, k=None, start_index=10, max_goals=10):
    def optimise_k(df,
               k_range=range(5, 100, 5),
               #*,
               start_index=0,
               max_goals=10,
               goal_target=2.5,
               n_workers=None):

        worker_fn = partial(_evaluate_k,
                            df,
                            start_index=start_index,
                            max_goals=max_goals,
                            goal_target=goal_target)

        best_LL = -float('inf')
        best_k  = None
        best_elohist = None
        best_eloratings = None

        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            futures = {exe.submit(worker_fn, k): k for k in k_range}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Optimising k"):
                total_LL, k, elohist, elorate = fut.result()
                if total_LL > best_LL:
                    best_LL, best_k, best_elohist, best_eloratings = total_LL, k, elohist, elorate

        print(f"Best k: {best_k} with LL: {best_LL:.2f}")
        return best_k, best_elohist, best_eloratings

    if k is None:
        k, elo_history, elo_ratings = optimise_k(df,start_index=start_index, max_goals=max_goals, goal_target=goal_target)
    else:
        elo_history, elo_ratings = get_elo(df, k=k)
        elo_history = elo_history[start_index:].reset_index(drop=True)

    scaler = StandardScaler()
    elo_history["EloDiff"] = scaler.fit_transform(elo_history["EloDiff"].values.reshape(-1, 1))

    home_model = glm("FTHG ~ EloDiff", data=elo_history,
                    family=sm.families.Poisson()).fit()
    away_model = glm("FTAG ~ EloDiff", data=elo_history,
                    family=sm.families.Poisson()).fit()

    return home_model, away_model, elo_ratings

def kelly_fraction(p, odds):
    # model p
    # market odds
    b = odds - 1
    q = 1 - p
    return (b * p - q) / b

if __name__ == "__main__":
    leagues = []
    for file in os.listdir("Data"):
        if file.endswith(".xlsx") or file.endswith(".csv"):
            leagues.append(file.split(".")[0])
    print(f"Available data: {', '.join(leagues)}")

    df = None

    while True:
        league = input("League: ").upper()
        if league in leagues:
            break
        print(f"League {league} not found")

    try:
        df = pd.read_excel(f"Data/{league}.xlsx")
    except:
        df = pd.read_csv(f"Data/{league}.csv")

    print(f"{len(df)} games loaded from {league}.")

    try:
        df = df[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
    except KeyError: # some of the leagues have mismatched names
        df.rename(columns={"HG": "FTHG", "AG": "FTAG"}, inplace=True)
        df.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam"}, inplace=True)

        df = df[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]

    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel("K"))

    print()
    team_dict = {i: team for i, team in enumerate(teams)}
    for i, team in team_dict.items():
        print(f"{i}: {team}")
    print()

    while True:
        try:
            home_team_index = int(input("Home team index: "))
            away_team_index = int(input("Away team index: "))

            if home_team_index == away_team_index:
                raise ValueError("Home and away teams cannot be the same.")
            if home_team_index not in team_dict or away_team_index not in team_dict:
                raise KeyError("Invalid index.")

            home_team = team_dict[home_team_index]
            away_team = team_dict[away_team_index]

        except ValueError:
            print("Invalid input. Please enter valid indices.")
            continue
        except KeyError:
            print("Invalid index. Please enter a valid index from the list.")
            continue

        break

    print(f"Selected teams: {home_team} vs {away_team}")

    while True:
        try:
            goal_target = float(input("Under/Over target: "))
            if goal_target <= 0:
                raise ValueError("Target must be a positive number.")
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            continue

    max_goals = 20 #int((df["FTHG"].max() + df["FTAG"].max()) * 1.5) # to ensure we cover all possible outcomes - actually this is a waste of compute for the tiny tails

    home_model, away_model, elo_ratings = fit_models(df, goal_target=goal_target, start_index=10, max_goals=max_goals)

    print(f"\n{home_team} model")
    print(home_model.summary())
    print(f"\n{away_team} model")
    print(away_model.summary())

    home_elo = elo_ratings[home_team]
    away_elo = elo_ratings[away_team]
    elo_diff = home_elo - away_elo

    home_pred = home_model.predict({"EloDiff": elo_diff})[0] # expected goals for home team
    away_pred = away_model.predict({"EloDiff": elo_diff})[0] # expected goals for away team

    p_total_goals = np.zeros((max_goals + 1, max_goals + 1))
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            if home_goals + away_goals > goal_target:
                p_total_goals[home_goals, away_goals] = poisson.pmf(home_goals, home_pred) * poisson.pmf(away_goals, away_pred)

    p_over = np.sum(p_total_goals)
    p_under = 1 - p_over

    under_odds = 1 / p_under
    over_odds = 1 / p_over

    print(f"\nPredicted goals: {home_pred:.2f} (Home), {away_pred:.2f} (Away)")
    print(f"Probability of under {goal_target}: {p_under:.4f} ({under_odds:.2f} odds)")
    print(f"Probability of over {goal_target}: {p_over:.4f} ({over_odds:.2f} odds)")
    
    while True:
        try:
            book_under_odds = float(input("\nBookmaker under odds: "))
            book_over_odds = float(input("Bookmaker over odds: "))
            
            if book_under_odds <= 0 or  book_over_odds <= 0:
                raise ValueError("Odds must be positive numbers.")
            break
        except ValueError:
            print("Invalid input. Please enter valid odds.")
            continue
    
    book_under_prob = 1 / book_under_odds
    book_over_prob = 1 / book_over_odds

    book_under_prob /= (book_under_prob + book_over_prob) # normalize to remove the bookmaker margin for fair comparison
    book_over_prob = 1 - book_under_prob
    #print(f"\nBookmaker normalized odds: Under {1/book_under_prob:.4f}, Over {1/book_over_prob:.4f}")

    under_edge = p_under - book_under_prob
    over_edge = p_over - book_over_prob

    print(f"\nEdge for under {goal_target}: {under_edge:.2%}")
    print(f"Edge for over {goal_target}: {over_edge:.2%}")

    kelly_under = kelly_fraction(p_under, book_under_odds)
    kelly_over = kelly_fraction(p_over, book_over_odds)

    print(f"\nKelly fraction for under {goal_target}: {kelly_under:.2%}")
    print(f"Kelly fraction for over {goal_target}: {kelly_over:.2%}")