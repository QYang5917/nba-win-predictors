"""Grade logged predictions against results and write predictions/SCORECARD.md.

Games: accuracy, log loss, and calibration (do 70% picks win ~70%?).
Props: average miss (MAE) of the model vs. the player's season average, and the over rate
(a good median prediction has about half of players scoring over it).
"""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PRED = ROOT / "data" / "raw", ROOT / "predictions"
GOAL = 1000


def grade_games() -> pd.DataFrame:
    g = pd.read_csv(PRED / "games.csv", dtype={"GAME_ID": str})
    tl = pd.read_csv(RAW / "team_game_logs_current.csv", dtype={"GAME_ID": str})
    tl["GAME_ID"] = tl["GAME_ID"].str.zfill(10)
    home = tl[tl["MATCHUP"].str.contains("vs.")].set_index("GAME_ID")
    g["GAME_ID"] = g["GAME_ID"].str.zfill(10)
    g["HOME_PTS"] = g["GAME_ID"].map(home["PTS"])
    g["AWAY_PTS"] = g["HOME_PTS"] - g["GAME_ID"].map(home["PLUS_MINUS"])
    g["HOME_WIN"] = (g["HOME_PTS"] > g["AWAY_PTS"]).where(g["HOME_PTS"].notna())
    g["CORRECT"] = ((g["P_HOME"] >= 0.5) == g["HOME_WIN"]).where(g["HOME_WIN"].notna())
    g.to_csv(PRED / "games.csv", index=False)
    return g


def grade_props() -> pd.DataFrame:
    p = pd.read_csv(PRED / "props.csv", dtype={"GAME_ID": str})
    logs = pd.read_csv(RAW / "player_game_logs_current.csv.gz", dtype={"GAME_ID": str})
    logs["GAME_ID"] = logs["GAME_ID"].str.zfill(10)
    p["GAME_ID"] = p["GAME_ID"].str.zfill(10)
    res = logs.set_index(["GAME_ID", "PLAYER_ID"])[["PTS", "MIN"]]
    idx = pd.MultiIndex.from_arrays([p["GAME_ID"], p["PLAYER_ID"]])
    p["ACTUAL_PTS"] = res["PTS"].reindex(idx).to_numpy()
    p["ACTUAL_MIN"] = res["MIN"].reindex(idx).to_numpy()
    p.to_csv(PRED / "props.csv", index=False)
    return p


def main():
    g, p = grade_games(), grade_props()
    done = g[g["HOME_WIN"].notna()]
    pr = p[p["ACTUAL_MIN"] > 0]  # a prop is void if the player doesn't play
    lines = [f"# Live prediction scorecard", "",
             f"Updated {dt.date.today()}. Model frozen on 2026-11-12; every prediction was logged before tip-off "
             f"(see the timestamps in `games.csv` and `props.csv`).", "",
             f"## Games: {len(done)} graded of {GOAL} goal ({len(g) - len(done)} pending)", ""]
    if len(done):
        q = done["P_HOME"].clip(1e-6, 1 - 1e-6)
        y = done["HOME_WIN"].astype(float)
        logloss = -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()
        conf = np.where(q >= 0.5, q, 1 - q)
        se = np.sqrt(done["CORRECT"].mean() * (1 - done["CORRECT"].mean()) / len(done))
        lines += [f"- **Accuracy: {done['CORRECT'].mean():.1%}** (±{1.96 * se:.1%} margin of error)",
                  f"- Always picking the home team: {y.mean():.1%}",
                  f"- Log loss: {logloss:.3f} (lower is better; a coin flip is 0.693)", "",
                  "| Model's confidence | Games | Picks won |", "|---|---|---|"]
        bins = pd.cut(conf, [0.5, 0.6, 0.7, 0.8, 1.0], include_lowest=True)
        for b, grp in done.groupby(bins, observed=True):
            lines.append(f"| {b.left:.0%}–{b.right:.0%} | {len(grp)} | {grp['CORRECT'].mean():.1%} |")
    lines += ["", f"## Player points: {len(pr)} graded", ""]
    if len(pr):
        mae_m = (pr["PRED_PTS"] - pr["ACTUAL_PTS"]).abs().mean()
        has_avg = pr["SEASON_AVG_PTS"].notna()
        mae_a = (pr.loc[has_avg, "SEASON_AVG_PTS"] - pr.loc[has_avg, "ACTUAL_PTS"]).abs().mean()
        mae_m2 = (pr.loc[has_avg, "PRED_PTS"] - pr.loc[has_avg, "ACTUAL_PTS"]).abs().mean()
        lines += [f"- **Average miss: model {mae_m2:.2f} pts vs season average {mae_a:.2f} pts** "
                  f"(same {has_avg.sum()} player-games)",
                  f"- Scored over the prediction: {(pr['ACTUAL_PTS'] > pr['PRED_PTS']).mean():.1%} "
                  f"(about 50% means the median is right)",
                  f"- Players who didn't play (void): {(p['ACTUAL_MIN'].isna() & p['GAME_ID'].isin(done['GAME_ID'])).sum()}"]
    lines += ["", "Betting lines aren't recorded yet, so these scores measure accuracy, not profit."]
    (PRED / "SCORECARD.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
