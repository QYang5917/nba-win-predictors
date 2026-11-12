"""Moneyball step 2: rate every player by how much they help their team win.

Method: RAPM with a box-score prior
-----------------------------------
1. **RAPM (Regularized Adjusted Plus-Minus).** For every game, the final margin is
   explained by who played and for how long:
       margin = home edge + sum(minutes/48 * rating, home players)
                          - sum(minutes/48 * rating, away players)
   Solving this across thousands of games separates each player's impact from his
   teammates' and opponents'. Ridge regression ("regularized") stops players with
   few minutes from getting extreme values.

2. **Box-score prior.** RAPM alone is noisy. So first we learn, from past seasons,
   how box-score stats (points, shots, turnovers, steals...) relate to RAPM. That gives
   every player a "first guess" from his stats. RAPM then starts from that guess and
   moves away from it only as far as the game results justify.

RATING is in points per 48 minutes vs. an average player. A team's expected margin
per game = sum over players of RATING * (minutes per game / 48).

For the current season, ratings use **last season + this season so far**.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LinearRegression, Ridge

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

STATS = ["PTS", "FGA", "FTA", "FG3M", "OREB", "DREB", "AST", "TOV", "STL", "BLK", "PF"]
LAMBDA = 30            # ridge strength, chosen by testing on held-out games
BOX_SHRINK_MIN = 300   # small-sample box scores are pulled toward the replacement level
REPLACEMENT = -2.0     # rating for unknown / fringe players


def load_logs() -> pd.DataFrame:
    df = pd.read_csv(RAW / "player_game_logs.csv.gz", dtype={"GAME_ID": str})
    return df[df["MIN"] > 0]


def league_per36(logs: pd.DataFrame) -> pd.DataFrame:
    tot = logs.groupby("SEASON")[STATS + ["MIN"]].sum()
    return tot[STATS].div(tot["MIN"], axis=0) * 36


def box_features(sub: pd.DataFrame, league: pd.DataFrame):
    """Per-36 stats relative to the league average of the seasons each player played."""
    tot = sub.groupby("PLAYER_ID")[STATS + ["MIN"]].sum()
    sm = sub.groupby(["PLAYER_ID", "SEASON"])["MIN"].sum().unstack(fill_value=0)
    lg = sm.div(sm.sum(axis=1), axis=0) @ league.loc[sm.columns]
    x = tot[STATS].div(tot["MIN"], axis=0) * 36 - lg.loc[tot.index]
    return x, tot["MIN"]


def design(sub: pd.DataFrame):
    """One row per game (from the lower TEAM_ID's side); columns = players."""
    g = sub.groupby(["GAME_ID", "TEAM_ID"], as_index=False).agg(
        PTS=("PTS", "sum"), HOME=("MATCHUP", lambda m: int("vs." in m.iloc[0])))
    g = g.sort_values(["GAME_ID", "TEAM_ID"])
    a = g.groupby("GAME_ID").nth(0).set_index("GAME_ID")
    b = g.groupby("GAME_ID").nth(1).set_index("GAME_ID")
    games = a.index.intersection(b.index)
    a, b = a.loc[games], b.loc[games]
    y = (a["PTS"] - b["PTS"]).to_numpy(float)
    h = (a["HOME"] - b["HOME"]).to_numpy(float)
    players = np.sort(sub["PLAYER_ID"].unique())
    gi = pd.Series(np.arange(len(games)), index=games)
    pi = pd.Series(np.arange(len(players)), index=players)
    s = sub[sub["GAME_ID"].isin(games)]
    sign = np.where(s["TEAM_ID"].to_numpy() == a.loc[s["GAME_ID"], "TEAM_ID"].to_numpy(), 1.0, -1.0)
    X = sparse.csr_matrix((sign * s["MIN"].to_numpy() / 48,
                           (gi[s["GAME_ID"]].to_numpy(), pi[s["PLAYER_ID"]].to_numpy())),
                          shape=(len(games), len(players)))
    return X, y, h, players


def rapm(sub: pd.DataFrame, prior: pd.Series | None = None, lam: float = LAMBDA):
    X, y, h, players = design(sub)
    home = (y * h).sum() / (h * h).sum()
    p = np.zeros(len(players)) if prior is None else \
        prior.reindex(players).fillna(REPLACEMENT).to_numpy()
    resid = y - home * h - X @ p
    coef = Ridge(alpha=lam, fit_intercept=False).fit(X, resid).coef_
    return pd.Series(coef + p, index=players), home


def fit_box_weights(logs: pd.DataFrame, seasons: list[str]) -> LinearRegression:
    """Learn box score -> RAPM from consecutive two-season windows in `seasons`."""
    league = league_per36(logs)
    seasons = sorted(seasons)
    rows = []
    for i in range(0, len(seasons) - 1, 2):
        sub = logs[logs["SEASON"].isin(seasons[i:i + 2])]
        r, _ = rapm(sub, lam=10)
        x, mins = box_features(sub, league)
        rows.append(x.assign(RAPM=r, MIN=mins))
    d = pd.concat(rows)
    d = d[d["MIN"] > 200]
    return LinearRegression().fit(d[STATS], d["RAPM"], sample_weight=d["MIN"])


def box_rating(sub: pd.DataFrame, box_model: LinearRegression, league: pd.DataFrame) -> pd.Series:
    x, mins = box_features(sub, league)
    raw = pd.Series(box_model.predict(x[STATS]), index=x.index)
    return (raw * mins + REPLACEMENT * BOX_SHRINK_MIN) / (mins + BOX_SHRINK_MIN)


def player_ratings(logs: pd.DataFrame, rating_seasons, box_model, cutoff_date=None) -> pd.DataFrame:
    """Final ratings from games in `rating_seasons` (optionally only before `cutoff_date`)."""
    sub = logs[logs["SEASON"].isin(rating_seasons)]
    if cutoff_date is not None:
        sub = sub[sub["GAME_DATE"] < cutoff_date]
    league = league_per36(logs)
    prior = box_rating(sub, box_model, league)
    rating, home = rapm(sub, prior)
    info = sub.groupby("PLAYER_ID").agg(PLAYER_NAME=("PLAYER_NAME", "last"),
                                        GP=("GAME_ID", "nunique"), MIN=("MIN", "sum"))
    out = info.assign(BOX_RATING=prior, RATING=rating).reset_index()
    out.attrs["home_edge"] = home
    return out.sort_values("RATING", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    logs = load_logs()
    history = sorted(s for s in logs["SEASON"].unique() if s != "2026-27")
    box_model = fit_box_weights(logs, history)
    print("Box-score prior: points per 48 added by +1 of each stat per 36 min")
    print(pd.Series(box_model.coef_, index=STATS).round(3).to_string())

    r = player_ratings(logs, ["2025-26", "2026-27"], box_model)
    out = ROOT / "data" / "processed" / "player_ratings_current.csv"
    r.to_csv(out, index=False)
    print(f"\nHome-court edge: {r.attrs['home_edge']:.2f} points")
    print(f"Top 20 players (2025-26 + 2026-27 to date, 1500+ min), saved {len(r)} to {out.name}:")
    print(r[r["MIN"] >= 1500].head(20)[["PLAYER_NAME", "GP", "MIN", "BOX_RATING", "RATING"]]
          .round(1).to_string(index=False))
