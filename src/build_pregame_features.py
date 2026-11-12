"""Part 1b: pregame features. Only information known BEFORE tip-off.

Why this exists: the Part 1 features (same-game shooting, turnovers, rebounds, free throws)
rebuild a team's points per possession with ~98% accuracy, so they almost reveal the
final score. They answer "what explains a win", not "who will win". This file builds a
fair forecasting version.

For every game, each team's stats are its averages from *earlier games that season*,
blended with last season's averages early on (when few games have been played):
    blended = (g * this_season + 10 * last_season) / (g + 10)
Features are home minus away, plus rest days.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "team_game_logs.csv"
OUT = ROOT / "data" / "processed" / "pregame_features.csv"
PRIOR_GAMES = 10
COMP = ["PTS", "FGM", "FG3M", "FG3A", "FGA", "FTM", "FTA", "OREB", "DREB", "TOV", "POSS"]


def rates(t: pd.DataFrame, p: str = "") -> pd.DataFrame:
    """Four Factors (offense and defense) and net rating from summed components."""
    return pd.DataFrame({
        "EFG": (t[p + "FGM"] + 0.5 * t[p + "FG3M"]) / t[p + "FGA"],
        "TOV_PCT": t[p + "TOV"] / t[p + "POSS"],
        "OREB_PCT": t[p + "OREB"] / (t[p + "OREB"] + t[p + "OPP_DREB"]),
        "FT_RATE": t[p + "FTM"] / t[p + "FGA"],
        "OPP_EFG": (t[p + "OPP_FGM"] + 0.5 * t[p + "OPP_FG3M"]) / t[p + "OPP_FGA"],
        "OPP_TOV_PCT": t[p + "OPP_TOV"] / t[p + "OPP_POSS"],
        "DREB_PCT": t[p + "DREB"] / (t[p + "DREB"] + t[p + "OPP_OREB"]),
        "OPP_FT_RATE": t[p + "OPP_FTM"] / t[p + "OPP_FGA"],
        "NET_RTG": 100 * (t[p + "PTS"] - t[p + "OPP_PTS"]) / t[p + "POSS"],
        "WIN_PCT": t[p + "W"] / t[p + "G"],
        # Part 1c extras (3-point tests); not used by the main pregame model
        "FG3A_RATE": t[p + "FG3A"] / t[p + "FGA"],
        "FG3_PCT": t[p + "FG3M"] / t[p + "FG3A"],
        "OPP_FG3A_RATE": t[p + "OPP_FG3A"] / t[p + "OPP_FGA"],
        "OPP_FG3_PCT": t[p + "OPP_FG3M"] / t[p + "OPP_FG3A"],
        "FG3M_NET_PER100": 100 * (t[p + "FG3M"] / t[p + "POSS"] - t[p + "OPP_FG3M"] / t[p + "OPP_POSS"]),
    })


def main() -> None:
    df = pd.read_csv(RAW, dtype={"GAME_ID": str})
    df["POSS"] = df["FGA"] + 0.44 * df["FTA"] - df["OREB"] + df["TOV"]
    df["W"] = (df["WL"] == "W").astype(int)
    df["G"] = 1
    opp = df[["GAME_ID", "TEAM_ID"] + COMP].rename(columns=lambda c: c if c == "GAME_ID" else "OPP_" + c)
    df = df.merge(opp, on="GAME_ID")
    df = df[df["TEAM_ID"] != df["OPP_TEAM_ID"]].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ID", "SEASON", "GAME_DATE"])

    sums = COMP + ["OPP_" + c for c in COMP] + ["W", "G"]
    # season-to-date totals BEFORE this game (shift so the current game is excluded)
    cum = df.groupby(["TEAM_ID", "SEASON"])[sums].cumsum() - df[sums]
    cur = rates(cum)
    g = cum["G"]

    # last season's full-season rates for the same team
    season_tot = df.groupby(["TEAM_ID", "SEASON"])[sums].sum()
    prev_rates = rates(season_tot)
    prev_rates.index = prev_rates.index.set_levels(
        [f"{int(s[:4]) + 1}-{str(int(s[:4]) + 2)[-2:]}" for s in prev_rates.index.levels[1]], level=1)
    prev = prev_rates.reindex(pd.MultiIndex.from_arrays([df["TEAM_ID"], df["SEASON"]])).set_index(df.index)
    league = rates(df[sums].sum().to_frame().T).iloc[0]  # fallback for the first season
    prev = prev.fillna(league)

    blended = (cur.fillna(0).mul(g, axis=0) + prev * PRIOR_GAMES).div(g + PRIOR_GAMES, axis=0)
    blended["REST_DAYS"] = df.groupby(["TEAM_ID", "SEASON"])["GAME_DATE"].diff().dt.days.clip(upper=5).fillna(5)
    blended["IS_HOME"] = df["MATCHUP"].str.contains("vs.").astype(int)
    blended[["GAME_ID", "SEASON", "GAME_DATE", "TEAM", "W"]] = \
        df[["GAME_ID", "SEASON", "GAME_DATE", "TEAM_ABBREVIATION", "W"]]

    b = blended.sort_values(["GAME_ID", "IS_HOME", "TEAM"], ascending=[True, False, True])
    home = b.groupby("GAME_ID").nth(0).set_index("GAME_ID")
    away = b.groupby("GAME_ID").nth(1).set_index("GAME_ID")
    feat_cols = list(rates(cum.head(1)).columns) + ["REST_DAYS"]
    out = pd.DataFrame({"SEASON": home["SEASON"], "GAME_DATE": home["GAME_DATE"],
                        "TEAM": home["TEAM"], "OPP": away["TEAM"]})
    for c in feat_cols:
        out[f"{c}_DIFF"] = home[c] - away[c]
    out["WIN"] = home["W"]
    out = out.reset_index().sort_values(["GAME_DATE", "GAME_ID"])
    out.to_csv(OUT, index=False)
    print(f"Saved {len(out)} games x {len(feat_cols)} pregame features to {OUT.name}")


if __name__ == "__main__":
    main()
