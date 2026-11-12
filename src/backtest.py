"""Moneyball step 3: does the method work? Replay past seasons as if it were "today".

For each season 2017-18 to 2025-26:
  - stop the clock 22 days into the season (same point as today, Nov 11, in 2026-27)
  - learn box-score weights and the age curve only from seasons *before* it (no peeking)
  - rate players on this season so far + last season (+ the one before at half weight)
  - project every team's final wins, then compare with what actually happened.

Baselines to beat:
  - "Last season": every team wins at last season's rate
  - "Record so far": every team keeps its current winning percentage
  - "Blend": a simple average of the two
"""
from pathlib import Path

import numpy as np
import pandas as pd

from project import expected_minutes, expected_wins, prev_season, season_start, team_strength
from ratings import fit_age_curve, fit_box_weights, load_ages, load_logs, season_ratings

ROOT = Path(__file__).resolve().parents[1]
DAYS_IN = 22


def team_games(logs: pd.DataFrame, season: str) -> pd.DataFrame:
    """One row per game: date, home team, away team, home win."""
    g = logs[logs["SEASON"] == season].groupby(["GAME_ID", "TEAM_ID"], as_index=False).agg(
        GAME_DATE=("GAME_DATE", "first"), MATCHUP=("MATCHUP", "first"), WL=("WL", "first"))
    g["HOME"] = g["MATCHUP"].str.contains("vs.")
    g = g.sort_values(["GAME_ID", "HOME"], ascending=[True, False])
    h = g.groupby("GAME_ID").nth(0).set_index("GAME_ID")
    a = g.groupby("GAME_ID").nth(1).set_index("GAME_ID")
    return pd.DataFrame({"GAME_DATE": h["GAME_DATE"], "HOME_ID": h["TEAM_ID"],
                         "AWAY_ID": a.loc[h.index, "TEAM_ID"], "HOME_WIN": h["WL"] == "W"})


def wins_table(games: pd.DataFrame) -> pd.Series:
    w = pd.concat([games.loc[games["HOME_WIN"], "HOME_ID"], games.loc[~games["HOME_WIN"], "AWAY_ID"]])
    teams = pd.concat([games["HOME_ID"], games["AWAY_ID"]]).unique()
    return w.value_counts().reindex(teams, fill_value=0)


def games_played(games: pd.DataFrame) -> pd.Series:
    return pd.concat([games["HOME_ID"], games["AWAY_ID"]]).value_counts()


def backtest_season(logs, ages, season):
    history = sorted(s for s in logs["SEASON"].unique() if s < season)
    box_model = fit_box_weights(logs, history)
    cutoff = (season_start(logs, season) + pd.Timedelta(days=DAYS_IN)).strftime("%Y-%m-%d")

    curve = fit_age_curve(logs, ages, box_model, season)
    ratings = season_ratings(logs, season, box_model, curve, ages, cutoff)
    minutes = expected_minutes(logs, season, cutoff)
    strength = team_strength(minutes, ratings)

    games = team_games(logs, season)
    before, after = games[games["GAME_DATE"] < cutoff], games[games["GAME_DATE"] >= cutoff]
    proj = expected_wins(strength, after, wins_table(before), ratings.attrs["home_edge"])

    n_games = games_played(games)
    actual = wins_table(games)
    prev_games = team_games(logs, prev_season(season))
    prev_pct = wins_table(prev_games) / games_played(prev_games)
    so_far_pct = (wins_table(before) / games_played(before)).fillna(0.5)

    out = pd.DataFrame({
        "SEASON": season,
        "ACTUAL": actual,
        "MODEL": proj["PROJ_WINS"],
        "LAST_SEASON": prev_pct * n_games,
        "RECORD_SO_FAR": so_far_pct * n_games,
        "N_GAMES": n_games,
    })
    out["BLEND"] = (out["LAST_SEASON"] + out["RECORD_SO_FAR"]) / 2
    return out


def main():
    logs = load_logs()
    ages = load_ages()
    seasons = [f"{y}-{str(y + 1)[-2:]}" for y in range(2017, 2026)]
    res = pd.concat([backtest_season(logs, ages, s) for s in seasons]).reset_index(names="TEAM_ID")
    # Compare on an 82-game scale so shortened seasons count fairly.
    methods = ["MODEL", "LAST_SEASON", "RECORD_SO_FAR", "BLEND"]
    scaled = res[methods + ["ACTUAL"]].div(res["N_GAMES"], axis=0) * 82
    err = scaled[methods].sub(scaled["ACTUAL"], axis=0)
    summary = pd.DataFrame({"MAE (wins)": err.abs().mean(), "RMSE (wins)": np.sqrt((err ** 2).mean()),
                            "Correlation": scaled[methods].corrwith(scaled["ACTUAL"])})
    by_season = err.abs().groupby(res["SEASON"]).mean()

    out = ROOT / "data" / "processed" / "backtest_results.csv"
    res.to_csv(out, index=False)
    print("Error in final wins (82-game scale), 270 team-seasons:")
    print(summary.round(2).to_string())
    print("\nMean absolute error by season:")
    print(by_season.round(1).to_string())


if __name__ == "__main__":
    main()
