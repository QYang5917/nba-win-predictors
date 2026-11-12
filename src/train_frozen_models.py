"""Train and FREEZE the two models used for live predictions (run once, before tracking starts).

Both are trained on every completed season (2016-17 to 2025-26) and then never changed
during the live test, so the 2026-27 results are a true out-of-sample check.

  models/game_model.json        P(home win) from player-based strength gap + rest days
  models/points_model.joblib    median points for a player in a game (props are over/under the median)
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from build_player_points import FEATURES, build_features, load_team_logs
from ratings import load_logs

ROOT = Path(__file__).resolve().parents[1]
PROC, MODELS = ROOT / "data" / "processed", ROOT / "models"
TRAIN_THROUGH = "2025-26"


def main():
    MODELS.mkdir(exist_ok=True)

    # Game model: logistic regression on unscaled features, so it's easy to apply by hand.
    ps = pd.read_csv(PROC / "player_strength_features.csv")
    pre = pd.read_csv(PROC / "pregame_features.csv")[["GAME_ID", "REST_DAYS_DIFF", "WIN"]]
    d = ps.merge(pre, on="GAME_ID")
    d = d[d["SEASON"] <= TRAIN_THROUGH]
    cols = ["PLAYER_STRENGTH_DIFF", "REST_DAYS_DIFF"]
    lr = LogisticRegression(max_iter=2000).fit(d[cols], d["WIN"])
    game = {"features": cols, "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
            "trained_on": f"{d['SEASON'].min()} to {d['SEASON'].max()}", "games": int(len(d)),
            "train_accuracy": round(float(lr.score(d[cols], d["WIN"])), 4)}
    (MODELS / "game_model.json").write_text(json.dumps(game, indent=2))
    print("Game model:", game)

    # Points model: gradient boosting that predicts the median (absolute-error loss).
    feats = build_features(load_logs(), load_team_logs())
    tr = feats[(feats["SEASON"] <= TRAIN_THROUGH) & ~feats["FUTURE"]]
    gb = HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.05,
                                       min_samples_leaf=200, random_state=42).fit(tr[FEATURES], tr["PTS"])
    joblib.dump({"model": gb, "features": FEATURES, "trained_on": f"{tr['SEASON'].min()} to {TRAIN_THROUGH}",
                 "rows": len(tr)}, MODELS / "points_model.joblib")
    print(f"Points model: {len(tr)} player-games, train MAE {(gb.predict(tr[FEATURES]) - tr['PTS']).abs().mean():.3f}")


if __name__ == "__main__":
    main()
