"""Research: can a richer points model beat the frozen one? (Run by hand; the live run does not use it.)

Set-up, fixed before looking at results:
  - train on 2016-17 to 2022-23, compare ideas on 2023-24 and 2024-25 (the "practice" seasons)
  - 2025-26 is held back and only scored once, for the version chosen on the practice seasons
  - scored on players expected to play 20+ minutes, like the live predictions
  - score = average miss in points (lower is better); the model predicts the median

Result (2026-11-19). Average miss in points:

                                        2023-24   2024-25   2025-26 (held back)
  season average                         5.322     5.368
  frozen model's features                5.078     5.089     5.045
  + minutes detail                       5.070     5.086
  + team minutes squeeze                 5.069     5.082
  + recent scoring                       5.072     5.092
  + shooting skill                       5.078     5.084
  + absence                              5.077     5.087
  + game context                         5.078     5.088
  + age                                  5.078     5.089
  + everything (36 new features)         5.054     5.075     5.028
  if real minutes were known             4.380     4.384

All 36 new features together gain about 0.02 points (0.3%), and the held-back season agrees
(0.017, give or take 0.010). Model settings (more trees, leaf size, training only on rotation
players) changed nothing. Knowing the real minutes would be worth 0.70 points, but minutes are
decided during the game (fouls, blowouts, injuries, the coach). A spread that depends on the player
did not improve the over/under chances either (log loss 0.5765 against 0.5759 for one table).

Conclusion: box scores are used up. A v2 built from them is not worth running. Any further gain has
to come from information that is not in box scores, such as confirmed starters or the betting line.

  python points_v2_research.py            compare feature groups on the practice seasons
  python points_v2_research.py final      score "everything" once on the held-back season
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from build_player_points import FEATURES, build_features, load_team_logs, team_context
from ratings import load_ages, load_logs

TRAIN_THROUGH, PRACTICE, HELD_BACK = "2022-23", ["2023-24", "2024-25"], "2025-26"
MIN_MINUTES = 20

GROUPS = {
    "minutes detail": ["MIN_LAST", "MIN_L3", "MIN_L5", "MIN_SD10", "MIN_SEASON", "PREV_MPG"],
    "team minutes squeeze": ["TEAM_EXP_MIN", "EXP_MIN_SHARE", "EXP_MIN_RANK", "EXP_MIN_SCALED"],
    "recent scoring": ["PTS_L3", "PTS_L5", "PTS_EWM20", "PTS_PM_RECENT", "USG_EWM", "FGA_EWM", "FTA_EWM", "FG3A_EWM"],
    "shooting skill": ["TS_SEASON", "FG3_PCT", "FT_PCT", "PREV_PPG"],
    "absence": ["DAYS_SINCE", "TEAM_GAMES_MISSED"],
    "game context": ["TEAM_DRTG", "OPP_ORTG", "NET_DIFF", "EXP_TOTAL", "B2B", "OPP_B2B", "TEAM_GAME_NO"],
    "age": ["AGE"],
}


def roll(g, n, how="mean"):
    """Value over the previous n games (this game not included)."""
    return g.transform(lambda x: getattr(x.shift().rolling(n, min_periods=1), how)())


def extra_features(logs: pd.DataFrame, feats: pd.DataFrame, team_logs: pd.DataFrame) -> pd.DataFrame:
    """New pregame features for every player-game in `feats` (rows where the player played)."""
    d = logs[logs["MIN"] > 0].copy()
    d["GAME_ID"] = d["GAME_ID"].str.zfill(10)
    d["GAME_DATE"] = pd.to_datetime(d["GAME_DATE"])
    d = d.sort_values(["PLAYER_ID", "SEASON", "GAME_DATE"]).reset_index(drop=True)
    d["USG"] = d.FGA + 0.44 * d.FTA + d.TOV
    key = ["PLAYER_ID", "SEASON"]
    g = d.groupby(key)

    # minutes
    d["MIN_LAST"] = g["MIN"].shift()
    d["MIN_L3"], d["MIN_L5"] = roll(g["MIN"], 3), roll(g["MIN"], 5)
    d["MIN_SD10"] = g["MIN"].transform(lambda x: x.shift().rolling(10, min_periods=3).std())
    d["MIN_SEASON"] = (g["MIN"].cumsum() - d["MIN"]) / g.cumcount().replace(0, np.nan)

    # recent scoring and shot volume
    d["PTS_L3"], d["PTS_L5"] = roll(g["PTS"], 3), roll(g["PTS"], 5)
    ewm = lambda c, span: g[c].transform(lambda x: x.ewm(span=span).mean().shift())
    d["PTS_EWM20"] = ewm("PTS", 20)
    d["PTS_PM_RECENT"] = ewm("PTS", 10) / ewm("MIN", 10)
    for c in ["USG", "FGA", "FTA", "FG3A"]:
        d[c + "_EWM"] = ewm(c, 10)

    # shooting skill: this season so far, pulled toward last season (or the league) when the sample is small
    tot = d.groupby(key)[["PTS", "FGA", "FTA", "FG3M", "FG3A", "FTM", "MIN"]].sum()
    gp = d.groupby(key).size()
    nxt = lambda s: f"{int(s[:4]) + 1}-{str(int(s[:4]) + 2)[-2:]}"
    prev = tot.assign(GP=gp)
    prev.index = prev.index.set_levels([nxt(s) for s in prev.index.levels[1]], level=1)
    p = prev.reindex(pd.MultiIndex.from_arrays([d.PLAYER_ID, d.SEASON])).reset_index(drop=True)
    cum = lambda c: g[c].cumsum() - d[c]
    lg = tot.sum()

    def shrunk(num_now, den_now, num_prev, den_prev, league, weight):
        prior = (num_prev.fillna(0) + league * weight) / (den_prev.fillna(0) + weight)  # last season, pulled to league
        return (num_now + prior * weight) / (den_now + weight)

    shots_now, shots_prev = 2 * (cum("FGA") + 0.44 * cum("FTA")), 2 * (p.FGA + 0.44 * p.FTA)
    d["TS_SEASON"] = shrunk(cum("PTS"), shots_now, p.PTS, shots_prev, lg.PTS / (2 * (lg.FGA + 0.44 * lg.FTA)), 300)
    d["FG3_PCT"] = shrunk(cum("FG3M"), cum("FG3A"), p.FG3M, p.FG3A, lg.FG3M / lg.FG3A, 150)
    d["FT_PCT"] = shrunk(cum("FTM"), cum("FTA"), p.FTM, p.FTA, lg.FTM / lg.FTA, 80)
    d["PREV_PPG"], d["PREV_MPG"] = p.PTS / p.GP, p.MIN / p.GP

    # absence: days since his last game, and team games he missed since then
    d["DAYS_SINCE"] = g["GAME_DATE"].diff().dt.days.clip(upper=30)
    tg = d[["TEAM_ID", "SEASON", "GAME_ID", "GAME_DATE"]].drop_duplicates(["TEAM_ID", "GAME_ID"]).sort_values(["TEAM_ID", "SEASON", "GAME_DATE"])
    tg["TEAM_GAME_NO"] = tg.groupby(["TEAM_ID", "SEASON"]).cumcount()
    tg["TEAM_REST"] = tg.groupby(["TEAM_ID", "SEASON"])["GAME_DATE"].diff().dt.days
    d = d.merge(tg[["TEAM_ID", "GAME_ID", "TEAM_GAME_NO", "TEAM_REST"]], on=["TEAM_ID", "GAME_ID"])
    d = d.sort_values(["PLAYER_ID", "SEASON", "GAME_DATE"]).reset_index(drop=True)
    same_team = d.groupby(key)["TEAM_ID"].shift() == d["TEAM_ID"]
    d["TEAM_GAMES_MISSED"] = (d["TEAM_GAME_NO"] - d.groupby(key)["TEAM_GAME_NO"].shift() - 1).where(same_team).clip(0, 20)
    d["B2B"] = (d["TEAM_REST"] == 1).astype(int)

    ages = load_ages()[["PLAYER_ID", "SEASON", "AGE"]].drop_duplicates(["PLAYER_ID", "SEASON"])
    d = d.merge(ages, on=["PLAYER_ID", "SEASON"], how="left")
    out = d[["GAME_ID", "PLAYER_ID", "MIN", "TEAM_REST", *[c for cols in GROUPS.values() for c in cols if c in d]]]
    f = feats.merge(out, on=["GAME_ID", "PLAYER_ID"], how="left", suffixes=("", "_X"))

    # team minutes squeeze: do tonight's players add up to more or less than a full game (240 minutes)?
    tgrp = f.groupby(["GAME_ID", "TEAM_ID"])
    f["TEAM_EXP_MIN"] = tgrp["EXP_MIN"].transform("sum")
    f["EXP_MIN_SHARE"] = f["EXP_MIN"] / f["TEAM_EXP_MIN"]
    f["EXP_MIN_RANK"] = tgrp["EXP_MIN"].rank(ascending=False)
    f["EXP_MIN_SCALED"] = f["EXP_MIN_SHARE"] * 240

    # game context: both teams' quality, expected scoring, schedule
    ctx = team_context(team_logs)
    f = f.merge(ctx[["GAME_ID", "TEAM_ID", "DRTG"]].rename(columns={"DRTG": "TEAM_DRTG"}), on=["GAME_ID", "TEAM_ID"], how="left")
    f = f.merge(ctx[["GAME_ID", "TEAM_ID", "ORTG"]].rename(columns={"TEAM_ID": "OPP_ID", "ORTG": "OPP_ORTG"}), on=["GAME_ID", "OPP_ID"], how="left")
    f["NET_DIFF"] = (f.TEAM_ORTG - f.TEAM_DRTG) - (f.OPP_ORTG - f.OPP_DRTG)
    f["EXP_TOTAL"] = (f.TEAM_ORTG + f.OPP_DRTG + f.OPP_ORTG + f.TEAM_DRTG) / 2 * (f.TEAM_PACE + f.OPP_PACE) / 200
    opp_b2b = f.drop_duplicates(["GAME_ID", "TEAM_ID"]).set_index(["GAME_ID", "TEAM_ID"])["B2B"]
    f["OPP_B2B"] = opp_b2b.reindex(pd.MultiIndex.from_arrays([f.GAME_ID, f.OPP_ID])).to_numpy()
    return f


def fit(train, cols, **kw):
    params = dict(loss="absolute_error", max_iter=400, learning_rate=0.05, min_samples_leaf=200, random_state=42)
    return HistGradientBoostingRegressor(**{**params, **kw}).fit(train[cols], train["PTS"])


def score(model, cols, test) -> dict:
    out = {}
    for s, t in test.groupby("SEASON"):
        out[s] = (t["PTS"] - model.predict(t[cols])).abs().mean()
    return out


def load():
    warnings.filterwarnings("ignore")
    logs, team_logs = load_logs(), load_team_logs()
    f = build_features(logs, team_logs)
    f = extra_features(logs, f[~f["FUTURE"]], team_logs)
    return f[f["SEASON"] <= TRAIN_THROUGH], f[f["SEASON"].isin(PRACTICE + [HELD_BACK]) & (f["EXP_MIN"] >= MIN_MINUTES)]


def main(final: bool = False):
    train, test = load()
    everything = FEATURES + [c for cols in GROUPS.values() for c in cols]
    if final:
        held = test[test["SEASON"] == HELD_BACK]
        old = (held["PTS"] - fit(train, FEATURES).predict(held[FEATURES])).abs()
        new = (held["PTS"] - fit(train, everything).predict(held[everything])).abs()
        by_day = (old - new).groupby(held["GAME_DATE"]).mean()
        print(f"Held-back {HELD_BACK} ({len(held)} player-games): frozen features {old.mean():.3f}, everything "
              f"{new.mean():.3f}, gain {(old - new).mean():.3f} (give or take {2 * by_day.std() / np.sqrt(len(by_day)):.3f})")
        return
    practice = test[test["SEASON"].isin(PRACTICE)]
    print(f"train {len(train)} player-games, practice {len(practice)}")
    base = score(fit(train, FEATURES), FEATURES, practice)
    show = lambda name, s: print(f"{name:34s} " + "  ".join(f"{k} {v:.3f} ({v - base[k]:+.3f})" for k, v in s.items()))
    show("season average", {s: (t["PTS"] - t["SEASON_AVG_PTS"].fillna(t["EXP_PTS"])).abs().mean()
                            for s, t in practice.groupby("SEASON")})
    show("frozen model's features", base)
    for name, cols in GROUPS.items():
        show("+ " + name, score(fit(train, FEATURES + cols), FEATURES + cols, practice))
    show("+ everything", score(fit(train, everything), everything, practice))
    show("if real minutes were known", score(fit(train, FEATURES + ["MIN"]), FEATURES + ["MIN"], practice))


if __name__ == "__main__":
    main(final=len(sys.argv) > 1 and sys.argv[1] == "final")
