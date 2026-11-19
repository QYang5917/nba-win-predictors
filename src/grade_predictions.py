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


def wilson_interval(correct, total, z=1.959963984540054):
    """Binomial interval; unlike the Wald interval it works at 0/n and n/n."""
    if total <= 0 or not 0 <= correct <= total:
        raise ValueError('Invalid successes/trials')
    p = correct / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return max(0., center-half), min(1., center+half)


def score_games(predictions, team_logs):
    """Grade from BOTH teams' actual PTS; never infer scores from PLUS_MINUS."""
    g, tl = predictions.copy(), team_logs.copy()
    for d in [g, tl]:
        d['GAME_ID'] = d.GAME_ID.astype(str).str.zfill(10)
    if g.GAME_ID.duplicated().any() or tl.duplicated(['GAME_ID','TEAM_ABBREVIATION']).any():
        raise ValueError('Duplicate prediction or team result')
    if 'WL' in tl.columns:
        # A game still being played shows a partial score and no W/L yet. Only final games are graded.
        tl = tl[tl.WL.notna()]
    pts = tl.set_index(['GAME_ID','TEAM_ABBREVIATION']).PTS
    for side in ['HOME','AWAY']:
        idx = pd.MultiIndex.from_arrays([g.GAME_ID, g[side]])
        g[side+'_PTS'] = pts.reindex(idx).to_numpy()
    predicted = pd.to_datetime(g.PREDICTED_AT, utc=True, errors='coerce')
    tip = pd.to_datetime(g.TIP_UTC, utc=True, errors='coerce')
    valid = predicted.notna() & tip.notna() & (predicted < tip)
    valid &= np.isfinite(g.P_HOME) & g.P_HOME.between(0,1) & (g.HOME != g.AWAY)
    count = g.GAME_ID.map(tl.groupby('GAME_ID').size())
    complete = g[['HOME_PTS','AWAY_PTS']].notna().all(axis=1) & (count == 2)
    complete &= (g.HOME_PTS != g.AWAY_PTS)
    g['GRADE_STATUS'] = np.where(~valid, 'invalid_prediction', np.where(complete, 'graded', 'pending_or_incomplete'))
    g['HOME_WIN'] = (g.HOME_PTS > g.AWAY_PTS).where(valid & complete)
    g['CORRECT'] = ((g.P_HOME >= .5) == g.HOME_WIN).where(valid & complete)
    return g


def grade_games() -> pd.DataFrame:
    g = pd.read_csv(PRED / "games.csv", dtype={"GAME_ID": str})
    tl = pd.read_csv(RAW / "team_game_logs_current.csv", dtype={"GAME_ID": str})
    g = score_games(g, tl)
    g.to_csv(PRED / "games.csv", index=False)
    return g


def grade_props() -> pd.DataFrame:
    p = pd.read_csv(PRED / "props.csv", dtype={"GAME_ID": str})
    logs = pd.read_csv(RAW / "player_game_logs_current.csv.gz", dtype={"GAME_ID": str})
    logs["GAME_ID"] = logs["GAME_ID"].str.zfill(10)
    p["GAME_ID"] = p["GAME_ID"].str.zfill(10)
    # Only take player results from games that are final for both teams (W/L recorded).
    tl = pd.read_csv(RAW / "team_game_logs_current.csv", dtype={"GAME_ID": str})
    final = tl[tl["WL"].notna()].assign(GAME_ID=lambda x: x["GAME_ID"].str.zfill(10)).groupby("GAME_ID").size()
    logs = logs[logs["GAME_ID"].isin(final[final == 2].index)]
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
             f"Updated {dt.date.today()}. Model frozen on 2026-11-12. Game grading checks recorded prediction "
             f"times precede tip-off; timestamps alone do not independently prove publication or input availability.", "",
             f"## Games: {len(done)} graded of {GOAL} goal "
             f"({(g['GRADE_STATUS'] == 'pending_or_incomplete').sum()} pending)", ""]
    lines += [f"Invalid game predictions excluded: {(g['GRADE_STATUS'] == 'invalid_prediction').sum()}.", ""]
    if len(done):
        q = done["P_HOME"].clip(1e-6, 1 - 1e-6)
        y = done["HOME_WIN"].astype(float)
        logloss = -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()
        conf = np.where(q >= 0.5, q, 1 - q)
        low, high = wilson_interval(int(done['CORRECT'].sum()), len(done))
        lines += [f"- **Accuracy: {done['CORRECT'].mean():.1%}** (95% Wilson interval {low:.1%}–{high:.1%}; assumes independent trials)",
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
        if pr["MODEL"].nunique() > 1:  # the lineup fix (v1.1) is scored apart from the earlier predictions
            lines += ["", "| Version | Player-games | Average miss | Scored over the prediction |", "|---|---|---|---|"]
            for version, v in pr.groupby("MODEL", sort=True):
                lines.append(f"| {version} | {len(v)} | {(v['PRED_PTS'] - v['ACTUAL_PTS']).abs().mean():.2f} pts "
                             f"| {(v['ACTUAL_PTS'] > v['PRED_PTS']).mean():.1%} |")
        lines += ["", "Until November 19, 2026 the live run fed the points model too many players per team, which made "
                  "its predictions about 1 point too low. Predictions labelled v1.1 use the same frozen model with that fixed."]
    lines += ["", "The scores above measure accuracy, not profit. Profit is tracked under paper bets."]
    (PRED / "SCORECARD.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
