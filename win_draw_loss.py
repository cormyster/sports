import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import glm
from scipy.stats import poisson
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
#from pymer4.models import Lmer

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

def get_elo_history(df, k=20):
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
            "EloDiff": elo_diff,
        })
    
    return pd.DataFrame(elo_history), elo_ratings

def _evaluate_k(df, k, start_index=10, max_goals=6):

    elo_history, elo_ratings = get_elo_history(df, k=k)
    elo_history = elo_history[start_index:]
    
    home_model = glm("FTHG ~ EloDiff", data=elo_history,
                     family=sm.families.Poisson()).fit()
    away_model = glm("FTAG ~ EloDiff", data=elo_history,
                     family=sm.families.Poisson()).fit()

    total_LL = 0.0
    for _, row in elo_history.iterrows():
        elo_diff   = row["EloDiff"]
        mu_home    = home_model.predict({"EloDiff": elo_diff})[0]
        mu_away    = away_model.predict({"EloDiff": elo_diff})[0]

        prob_matrix = np.outer(
            poisson.pmf(range(max_goals+1), mu_home),
            poisson.pmf(range(max_goals+1), mu_away)
        )

        p_home = np.tril(prob_matrix, -1).sum()
        p_draw = np.trace(prob_matrix)
        p_away = np.triu(prob_matrix,  1).sum()

        if   row["FTHG"] > row["FTAG"]:
            p = p_home
        elif row["FTHG"] < row["FTAG"]:
            p = p_away
        else:
            p = p_draw

        total_LL += np.log(max(p, 1e-12))

    return total_LL, k, elo_history, elo_ratings

def fit_models(df,
               k=None,
               k_range=range(5, 100, 5),
               start_index=10,
               max_goals=6,
               n_workers=None):
    
    if k is None:
        worker_fn = partial(_evaluate_k,
                            df,
                            start_index=start_index,
                            max_goals=max_goals)

        best_LL, best_k, best_elohist = -np.inf, None, None
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            futures = {exe.submit(worker_fn, k): k for k in k_range}
            for fut in tqdm(as_completed(futures),
                            total=len(futures),
                            desc="Optimising K"):
                total_LL, this_k, elohist, elorate = fut.result()
                if total_LL > best_LL:
                    best_LL, best_k, best_elohist, best_eloratings = total_LL, this_k, elohist, elorate
        k = best_k
        elo_history = best_elohist
        elo_ratings = best_eloratings
        print(f"Best K: {k}  |  Log-likelihood: {best_LL:.2f}")
    else:
        elo_history, elo_ratings = get_elo_history(df, k=k)
        elo_history = elo_history[start_index:]

    home_glm = glm("FTHG ~ EloDiff", data=elo_history,
                   family=sm.families.Poisson()).fit()
    away_glm = glm("FTAG ~ EloDiff", data=elo_history,
                   family=sm.families.Poisson()).fit()

    return home_glm, away_glm, elo_ratings

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

    max_goals = 20 #int((df["FTHG"].max() + df["FTAG"].max()) * 1.5) # to ensure we cover all possible outcomes

    home_model, away_model, elo_ratings = fit_models(df)

    print(f"\n{home_team} model")
    print(home_model.summary())
    print(f"\n{away_team} model")
    print(away_model.summary())

    home_elo = elo_ratings[home_team]
    away_elo = elo_ratings[away_team]
    elo_diff = home_elo - away_elo
    mu_home = home_model.predict({"EloDiff": elo_diff})[0] # expected goals for home team
    mu_away = away_model.predict({"EloDiff": elo_diff})[0] # expected goals for away team

    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob_matrix[home_goals, away_goals] = poisson.pmf(home_goals, mu_home) * poisson.pmf(away_goals, mu_away)

    p_home = np.sum(np.tril(prob_matrix, -1)) 
    p_draw = np.sum(np.diag(prob_matrix))
    p_away = np.sum(np.triu(prob_matrix, 1))

    home_odds = 1 / p_home
    draw_odds = 1 / p_draw
    away_odds = 1 / p_away

    print(f"\nPredicted score: {mu_home:.2f} - {mu_away:.2f}")
    print(f"Home win probability: {p_home:.4f}, Odds: {home_odds:.2f}")
    print(f"Draw probability: {p_draw:.4f}, Odds: {draw_odds:.2f}")
    print(f"Away win probability: {p_away:.4f}, Odds: {away_odds:.2f}")

    while True:
        try:
            book_home_odds = float(input("\nBookmaker home odds: "))
            book_draw_odds = float(input("Bookmaker draw odds: "))
            book_away_odds = float(input("Bookmaker away odds: "))

            if book_home_odds <= 0 or book_draw_odds <= 0 or book_away_odds <= 0:
                raise ValueError("Odds must be positive numbers.")
            break
        except ValueError:
            print("Invalid input. Please enter valid odds.")
            continue

    book_probs = np.array([1 / book_home_odds, 1 / book_draw_odds, 1 / book_away_odds])
    book_probs /= np.sum(book_probs)
    book_odds = 1 / book_probs
    #print(f"\nBookmaker normalised odds: Home: {book_odds[0]:.2f}, Draw: {book_odds[1]:.2f}, Away: {book_odds[2]:.2f}")

    home_edge = p_home - book_probs[0]
    draw_edge = p_draw - book_probs[1]
    away_edge = p_away - book_probs[2]
    print(f"\nEdge: Home: {home_edge:.2%}, Draw: {draw_edge:.2%}, Away: {away_edge:.2%}")

    kelly_home = kelly_fraction(p_home, book_home_odds)
    kelly_draw = kelly_fraction(p_draw, book_draw_odds)
    kelly_away = kelly_fraction(p_away, book_away_odds)
    print(f"\nKelly fraction: Home: {kelly_home:.2%}, Draw: {kelly_draw:.2%}, Away: {kelly_away:.2%}")
