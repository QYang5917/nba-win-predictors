"""Step 2: Turn raw box scores into per-game features.

Output: one row per game, seen from the home team's side.
Every stat is "home team minus away team", so a positive number means the home
team did better in that area. Target: did the home team win (1/0)?

Points and plus-minus are deliberately left out: they *are* the result
(more points than the other team = a win), so including them would be cheating.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "team_game_logs.csv"
OUT = ROOT / "data" / "processed" / "game_features.csv"


def team_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-team, per-game efficiency stats (no opponent info needed)."""
    out = df.copy()
    # Possessions estimate (standard formula used by Basketball-Reference / NBA.com)
    out["POSS"] = out["FGA"] + 0.44 * out["FTA"] - out["OREB"] + out["TOV"]
    out["EFG_PCT"] = (out["FGM"] + 0.5 * out["FG3M"]) / out["FGA"]  # shooting, with 3s worth 1.5x
    out["TOV_PCT"] = out["TOV"] / out["POSS"]                         # share of possessions lost to turnovers
    out["FT_RATE"] = out["FTM"] / out["FGA"]                          # getting to (and making) free throws
    out["FG3A_RATE"] = out["FG3A"] / out["FGA"]                       # how 3-point-heavy the offense is
    out["AST_RATE"] = out["AST"] / out["FGM"]                         # share of baskets that were assisted
    out["STL_PER100"] = 100 * out["STL"] / out["POSS"]
    out["BLK_PER100"] = 100 * out["BLK"] / out["POSS"]
    out["IS_HOME"] = out["MATCHUP"].str.contains("vs.").astype(int)
    return out


def main() -> None:
    raw = pd.read_csv(RAW, dtype={"GAME_ID": str})
    df = team_rates(raw)

    # Pair each team with its opponent in the same game.
    opp = df[["GAME_ID", "TEAM_ID", "OREB", "DREB", "REB", "POSS"]].rename(
        columns=lambda c: c if c == "GAME_ID" else f"OPP_{c}"
    )
    df = df.merge(opp, on="GAME_ID")
    df = df[df["TEAM_ID"] != df["OPP_TEAM_ID"]]

    # Stats that need the opponent's numbers
    df["OREB_PCT"] = df["OREB"] / (df["OREB"] + df["OPP_DREB"])  # grabbing your own misses
    df["REB_MARGIN"] = df["REB"] - df["OPP_REB"]
    game_minutes = df["MIN"] / 5  # MIN is total player-minutes (5 players on court)
    df["PACE"] = 48 * ((df["POSS"] + df["OPP_POSS"]) / 2) / game_minutes  # possessions per 48 min

    # One row per game: the home team's view. (A handful of games have no
    # "vs." row in the data; for those, fall back to the lower TEAM_ID.)
    df = df.sort_values(["GAME_ID", "IS_HOME", "TEAM_ID"], ascending=[True, False, True])
    team = df.groupby("GAME_ID").nth(0).set_index("GAME_ID")
    away = df.groupby("GAME_ID").nth(1).set_index("GAME_ID")

    diff_cols = ["EFG_PCT", "TOV_PCT", "OREB_PCT", "FT_RATE", "FG3A_RATE",
                 "AST_RATE", "STL_PER100", "BLK_PER100", "IS_HOME"]
    feats = pd.DataFrame(index=team.index)
    feats["SEASON"] = team["SEASON"]
    feats["GAME_DATE"] = team["GAME_DATE"]
    feats["TEAM"] = team["TEAM_ABBREVIATION"]
    feats["OPP"] = away["TEAM_ABBREVIATION"]
    for c in diff_cols:
        feats[f"{c}_DIFF"] = team[c] - away[c]
    feats["REB_MARGIN"] = team["REB_MARGIN"]
    feats["PACE"] = team["PACE"]
    feats["WIN"] = (team["WL"] == "W").astype(int)

    feats = feats.rename(columns={"IS_HOME_DIFF": "HOME"}).reset_index()
    feats = feats.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUT, index=False)
    print(f"Saved {len(feats)} games x {feats.shape[1]} columns to {OUT}")


if __name__ == "__main__":
    main()
