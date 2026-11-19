"""Research: which lineup should the live run give the points model? (Run by hand; nothing else uses it.)

The points model was trained on the players who really played each game, about 11 a team. Before a
game nobody knows that list, so the live run has to guess it. Until 2026-11-19 it used every rostered
player not ruled out on the injury report, about 14 a team. This script replays past seasons the way
the live run works (tonight's rows built from a guessed lineup) and compares guesses.

A copy of the model is trained through 2022-23, so every season replayed here is unseen. Past
seasons have no saved injury reports, so a sitting player expected to play 12+ minutes is treated
as "reported out" and a sitting deep-bench player as unknown.

Result, every third game day, players expected to play 20+ minutes:

  2023-24 (4,911 player-games)        scored over the prediction   average miss
    whole roster (old live way)                53.5%                  5.18
    top 11 by expected minutes                 49.9%                  5.16
    true lineup (not knowable)                 49.1%                  5.14
  2024-25 (5,389)
    whole roster                               54.0%                  5.22
    top 11                                     51.0%                  5.21
    true lineup                                49.7%                  5.18
  2025-26 (1,981, every eighth game day)
    whole roster                               54.3%                  5.11
    top 11                                     51.1%                  5.09
    true lineup                                51.1%                  5.08

The whole roster makes predictions 0.6 to 0.8 points too low: more players share the team's shots,
and deep-bench players who sat last game look like "returning" players. The live predictions showed
the same thing (57% of 553 players beat the prediction). Every tighter guess fixes most of it and
they are within 0.01 points of each other in average miss. Top 11 was picked because it matches the
real lineup size and does not depend on how complete the injury report is: rules based on minutes
still ran low (52% to 53% over) when fewer absences were known. predict_today.py uses it (PROP_LINEUP).

  python lineup_replay.py 2025-26 8     season, and use every Nth game day (a full season takes a while)
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import build_player_strength as bps
from build_player_points import FEATURES, build_features, load_team_logs
from predict_today import STAT_COLS
from ratings import load_logs

TRAIN_THROUGH = "2022-23"
KNOWN_OUT_MIN = 12   # a sitting player expected to play this much is assumed to be on the injury report
MIN_MINUTES = 20     # same cut as the live predictions
top = lambda n: lambda c: c.sort_values("EXP", ascending=False).groupby("TEAM_ID").head(n)
RULES = {
    "whole roster not ruled out (old live way)": lambda c: c,
    "played last game, or expected 12+ min": lambda c: c[c.LAST | (c.EXP >= 12)],
    "expected 10+ min": lambda c: c[c.EXP >= 10],
    "top 10 by expected minutes": top(10),
    "top 11 by expected minutes (live now)": top(11),
    "true lineup (not knowable)": lambda c: c[c.PLAYS],
}


def candidates(today, last, exp, team_last_game) -> pd.DataFrame:
    """Every player the live run would consider for today's games, with what is known before tip-off."""
    out = []
    for (gid, tid), g in today.groupby(["GAME_ID", "TEAM_ID"]):
        ids = set(last.index[last.TEAM_ID == tid]) | (set(g.PLAYER_ID) & set(last.index))
        c = pd.DataFrame({"PLAYER_ID": sorted(ids)}).assign(GAME_ID=gid, TEAM_ID=tid)
        c["PLAYS"] = c.PLAYER_ID.isin(set(g.PLAYER_ID))
        c["EXP"] = c.PLAYER_ID.map(exp)
        c["LAST"] = (c.PLAYER_ID.map(last.GAME_ID) == team_last_game.get(tid)) & (c.PLAYER_ID.map(last.TEAM_ID) == tid)
        c["MATCHUP"], c["TEAM_ABBREVIATION"] = g.MATCHUP.iloc[0], g.TEAM_ABBREVIATION.iloc[0]
        out.append(c[c.PLAYS | (c.EXP < KNOWN_OUT_MIN)])
    return pd.concat(out)


def main(season: str, step: int):
    warnings.filterwarnings("ignore")
    logs = load_logs()
    logs["GAME_ID"] = logs["GAME_ID"].str.zfill(10)
    logs, team_logs = logs[logs["MIN"] > 0], load_team_logs()
    full = build_features(logs, team_logs)
    tr = full[full.SEASON <= TRAIN_THROUGH]
    model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.05,
                                          min_samples_leaf=200, random_state=42).fit(tr[FEATURES], tr.PTS)
    prev = f"{int(season[:4]) - 1}-{season[2:4]}"
    logs = logs[logs.SEASON.isin([prev, season])]
    team_logs = team_logs[team_logs.SEASON.isin([prev, season])]
    em = bps.expected_minutes(logs)
    zeros = {c: 0 for c in STAT_COLS}
    rows = []
    for day in sorted(logs.loc[logs.SEASON == season, "GAME_DATE"].unique())[::step]:
        before, today = logs[logs.GAME_DATE < day], logs[logs.GAME_DATE == day]
        cur = before[before.SEASON == season].sort_values("GAME_DATE")
        if cur.empty:
            continue
        last = cur.groupby("PLAYER_ID").last()  # each player's latest game this season
        exp = em[(em.SEASON == season) & (em.GAME_DATE < day)].sort_values("GAME_DATE").groupby("PLAYER_ID")["EXP_AFTER"].last()
        cands = candidates(today, last, exp, cur.groupby("TEAM_ID")["GAME_ID"].last())
        teams = today.drop_duplicates(["GAME_ID", "TEAM_ID"])[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "MATCHUP"]].assign(
            SEASON=season, GAME_DATE=day, WL=None, **zeros)
        actual = today.set_index(["GAME_ID", "PLAYER_ID"])["PTS"]
        for name, rule in RULES.items():
            tonight = rule(cands)[["PLAYER_ID", "GAME_ID", "TEAM_ID", "MATCHUP", "TEAM_ABBREVIATION"]].assign(
                SEASON=season, GAME_DATE=day, WL=None, FUTURE=True, PLAYER_NAME="", **zeros)
            f = build_features(pd.concat([before.assign(FUTURE=False), tonight], ignore_index=True),
                               pd.concat([team_logs[team_logs.GAME_DATE < day], teams], ignore_index=True))
            f = f[f.FUTURE & (f.EXP_MIN >= MIN_MINUTES)].copy()
            f["MISS"] = [actual.get(k, np.nan) for k in zip(f.GAME_ID, f.PLAYER_ID)] - model.predict(f[FEATURES])
            rows.append(f.assign(RULE=name)[["RULE", "MISS"]].dropna())
    r = pd.concat(rows)
    print(f"Season {season}, every {step} game day(s), players expected to play {MIN_MINUTES}+ minutes:")
    print(r.groupby("RULE", sort=False)["MISS"].agg(
        player_games="size", scored_over=lambda x: (x > 0).mean(), too_low_by="mean",
        average_miss=lambda x: x.abs().mean()).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2025-26", int(sys.argv[2]) if len(sys.argv) > 2 else 8)
