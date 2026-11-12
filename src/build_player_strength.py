"""Part 1d: pregame team strength built from PLAYERS, not team averages.

For every game, each team's strength = sum over its players of
    player rating (from ratings.py) * expected minutes / 48
using only information from before the game:
  - Player ratings are refit every 14 days, each time using only games before that date.
    Box-score weights and the age curve are learned only from earlier seasons.
  - Expected minutes = a player's recent minutes per game (weighted toward the last ~10
    games), blended with last season's average early on.
  - Only the top 240 expected minutes count (one game = 5 players x 48 minutes). Any
    shortfall is filled at replacement level.

Two versions of "who is available":
  LINEUP   players who actually played in this game. In real use this is known shortly
           before tip-off from the injury report and starting lineups. Small leak: a healthy
           player benched by the coach (DNP) is missing here, and the real injury report
           can change late.
  LAST_GAME  players who played in the team's previous game. Zero leakage, but misses
           players returning from injury and includes players who just got hurt.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ratings import (REPLACEMENT, fit_age_curve, fit_box_weights, load_ages, load_logs,
                     prev_season, season_ratings)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "player_strength_features.csv"
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2016, 2026)]
REFRESH_DAYS = 14
EWM_SPAN = 10          # recent-minutes window
PRIOR_GAMES = 5        # weight of last season's minutes per game early on
ROOKIE_MIN = 12.0      # expected minutes for a player with no history at all
TEAM_MIN = 240.0


def expected_minutes(logs: pd.DataFrame) -> pd.DataFrame:
    """For every player-game: expected minutes BEFORE the game and AFTER it (for the next game)."""
    d = logs.sort_values(["PLAYER_ID", "SEASON", "GAME_DATE"]).copy()
    grp = d.groupby(["PLAYER_ID", "SEASON"])["MIN"]
    d["EWM_AFTER"] = grp.transform(lambda s: s.ewm(span=EWM_SPAN).mean())
    d["EWM_BEFORE"] = d.groupby(["PLAYER_ID", "SEASON"])["EWM_AFTER"].shift()
    d["G_BEFORE"] = grp.cumcount()
    mpg = logs.groupby(["PLAYER_ID", "SEASON"])["MIN"].mean()
    nxt = {s: f"{int(s[:4]) + 1}-{str(int(s[:4]) + 2)[-2:]}" for s in mpg.index.levels[1]}
    mpg.index = mpg.index.set_levels([nxt[s] for s in mpg.index.levels[1]], level=1)
    prev = mpg.reindex(pd.MultiIndex.from_arrays([d["PLAYER_ID"], d["SEASON"]])).to_numpy()

    def blend(ewm, g):
        cur = np.nan_to_num(ewm.to_numpy(), nan=0.0)
        has_prev = ~np.isnan(prev)
        out = np.where(has_prev, (g * cur + PRIOR_GAMES * np.nan_to_num(prev)) / (g + PRIOR_GAMES), cur)
        return np.where((g == 0) & ~has_prev, ROOKIE_MIN, out)

    d["EXP_BEFORE"] = blend(d["EWM_BEFORE"], d["G_BEFORE"].to_numpy())
    d["EXP_AFTER"] = blend(d["EWM_AFTER"], d["G_BEFORE"].to_numpy() + 1)
    return d


def strength(rows: pd.DataFrame, exp_col: str) -> pd.Series:
    """rows: one per (GAME_ID, TEAM_ID, player) with RATING and an expected-minutes column."""
    r = rows.sort_values(exp_col, ascending=False)
    used = r.groupby(["GAME_ID", "TEAM_ID"])[exp_col].cumsum() - r[exp_col]
    take = np.clip(TEAM_MIN - used, 0, r[exp_col])
    s = (take * r["RATING"]).groupby([r["GAME_ID"], r["TEAM_ID"]]).sum()
    filled = take.groupby([r["GAME_ID"], r["TEAM_ID"]]).sum()
    return (s + (TEAM_MIN - filled) * REPLACEMENT) / 48


def season_box_model(logs, season):
    history = sorted(s for s in logs["SEASON"].unique() if s < season)
    if len(history) < 2:  # first season: only one earlier season to learn from
        history = history * 2
    return fit_box_weights(logs, history)


def main():
    logs = load_logs()
    ages = load_ages()
    pg = expected_minutes(logs)
    pg["HOME"] = pg["MATCHUP"].str.contains("vs.")

    # team-game order, for the LAST_GAME version
    tg = pg[["GAME_ID", "TEAM_ID", "SEASON", "GAME_DATE"]].drop_duplicates(["GAME_ID", "TEAM_ID"])
    tg = tg.sort_values(["TEAM_ID", "SEASON", "GAME_DATE"])
    tg["NEXT_GAME_ID"] = tg.groupby(["TEAM_ID", "SEASON"])["GAME_ID"].shift(-1)

    out = []
    for season in SEASONS:
        box = season_box_model(logs, season)
        early = sorted(s for s in logs["SEASON"].unique() if s < season)
        curve = fit_age_curve(logs, ages, box, season) if len(early) >= 2 else (lambda a: 0 * a)
        sp = pg[pg["SEASON"] == season].copy()
        start = pd.Timestamp(sp["GAME_DATE"].min())
        end = pd.Timestamp(sp["GAME_DATE"].max())
        cutoffs = pd.date_range(start, end, freq=f"{REFRESH_DAYS}D").strftime("%Y-%m-%d")
        ratings = {c: season_ratings(logs, season, box, curve, ages, c).set_index("PLAYER_ID")["RATING"]
                   for c in cutoffs}
        window = lambda dates: pd.Series(cutoffs[np.searchsorted(cutoffs, dates, side="right") - 1],
                                         index=dates.index)

        # LINEUP: players in this game, minutes expected before tip-off
        sp["CUTOFF"] = window(sp["GAME_DATE"])
        sp["RATING"] = [ratings[c].get(p, REPLACEMENT) for c, p in zip(sp["CUTOFF"], sp["PLAYER_ID"])]
        lineup = strength(sp, "EXP_BEFORE").rename("LINEUP")

        # LAST_GAME: players from the team's previous game, minutes expected after that game
        lg = sp.merge(tg[["GAME_ID", "TEAM_ID", "NEXT_GAME_ID"]], on=["GAME_ID", "TEAM_ID"])
        lg = lg.dropna(subset=["NEXT_GAME_ID"])
        nxt_date = tg.set_index(["GAME_ID", "TEAM_ID"])["GAME_DATE"]
        lg["GAME_ID"] = lg["NEXT_GAME_ID"]
        lg["CUTOFF"] = window(pd.Series(nxt_date.reindex(
            pd.MultiIndex.from_arrays([lg["GAME_ID"], lg["TEAM_ID"]])).to_numpy(), index=lg.index))
        lg["RATING"] = [ratings[c].get(p, REPLACEMENT) for c, p in zip(lg["CUTOFF"], lg["PLAYER_ID"])]
        last = strength(lg, "EXP_AFTER").rename("LAST_GAME")

        teams = sp.drop_duplicates(["GAME_ID", "TEAM_ID"]).set_index(["GAME_ID", "TEAM_ID"])[["HOME"]]
        st = teams.join(lineup).join(last).reset_index()
        home = st[st["HOME"]].set_index("GAME_ID")
        away = st[~st["HOME"]].set_index("GAME_ID")
        games = home.index.intersection(away.index)
        out.append(pd.DataFrame({
            "SEASON": season,
            "PLAYER_STRENGTH_DIFF": home.loc[games, "LINEUP"] - away.loc[games, "LINEUP"],
            # season openers have no previous game: treat as even
            "LAST_GAME_STRENGTH_DIFF": (home.loc[games, "LAST_GAME"] - away.loc[games, "LAST_GAME"]).fillna(0),
        }))
        print(season, len(games), "games,", len(cutoffs), "rating refreshes")

    res = pd.concat(out).reset_index(names="GAME_ID")
    res.to_csv(OUT, index=False)
    print(f"Saved {len(res)} games to {OUT.name}")


if __name__ == "__main__":
    main()
