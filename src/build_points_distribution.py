"""Build the table that turns a points prediction into an over/under chance (run once).

The frozen points model predicts a player's MEDIAN points. To price a bet we also need to know
how far real scores usually land from that median: "if we predict 18.2, how often does he score
more than 21.5?"

To answer honestly we need misses the model made on games it had never seen. So a copy of the
model (same settings) is trained on 2016-17 to 2022-23 and its misses are measured on 2023-24 to
2025-26. Misses are grouped by the size of the prediction, because stars miss by more than
role players. The frozen model itself is not touched.

  models/points_distribution.json   for each prediction range, the spread of (actual - predicted)
"""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from build_player_points import FEATURES, build_features, load_team_logs
from ratings import load_logs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "points_distribution.json"
TRAIN_THROUGH, CHECK_SEASON = "2022-23", "2025-26"
MIN_MINUTES = 20                       # live predictions are only logged for 20+ expected minutes
EDGES = [0, 8, 11, 14, 17, 21, 100]    # prediction ranges (points)
QUANTILES = np.round(np.arange(0.005, 1.0, 0.005), 3)


def table(d) -> list[dict]:
    """For each prediction range: quantiles of (actual - predicted)."""
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        r = d.loc[(d["PRED"] >= lo) & (d["PRED"] < hi), "RESID"]
        out.append({"lo": lo, "hi": hi, "n": int(len(r)), "resid": np.quantile(r, QUANTILES).round(3).tolist()})
    return out


def prob_over(bins: list[dict], pred: float, line: float) -> float:
    """Chance the player scores MORE than `line`, given the predicted median `pred`."""
    b = next((b for b in bins if b["lo"] <= pred < b["hi"]), bins[-1])
    # share of past misses that were bigger than (line - pred)
    return float(1 - np.interp(line - pred, b["resid"], QUANTILES, left=0.0, right=1.0))


def main():
    f = build_features(load_logs(), load_team_logs())
    f = f[~f["FUTURE"]]
    tr = f[f["SEASON"] <= TRAIN_THROUGH]
    te = f[(f["SEASON"] > TRAIN_THROUGH) & (f["SEASON"] <= "2025-26") & (f["EXP_MIN"] >= MIN_MINUTES)].copy()
    gb = HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.05,
                                       min_samples_leaf=200, random_state=42).fit(tr[FEATURES], tr["PTS"])
    te["PRED"] = gb.predict(te[FEATURES])
    te["RESID"] = te["PTS"] - te["PRED"]
    print(f"Unseen player-games: {len(te)}, average miss {te['RESID'].abs().mean():.2f}, "
          f"scored over the prediction {(te['RESID'] > 0).mean():.1%}")

    # Honesty check: build the table WITHOUT the last season, then see if its chances hold up there.
    early, late = te[te["SEASON"] < CHECK_SEASON], te[te["SEASON"] == CHECK_SEASON]
    bins = table(early)
    print(f"Check on {CHECK_SEASON} ({len(late)} player-games), table built from earlier seasons only:")
    for off in [-4.5, -2.5, -0.5, 1.5, 3.5]:
        line = (late["PRED"] + off).round() + 0.5  # a made-up line near the prediction, like a real x.5 line
        p = np.array([prob_over(bins, a, b) for a, b in zip(late["PRED"], line)])
        print(f"  line about {off:+.1f} from prediction: table says over {p.mean():.1%}, really over "
              f"{(late['PTS'] > line).mean():.1%}")

    out = {"built_from": f"model copy trained through {TRAIN_THROUGH}, misses on 2023-24 to 2025-26",
           "min_expected_minutes": MIN_MINUTES, "quantiles": QUANTILES.tolist(), "bins": table(te)}
    OUT.write_text(json.dumps(out))
    print("Saved", OUT.name, [(b["lo"], b["hi"], b["n"]) for b in out["bins"]])


if __name__ == "__main__":
    main()
