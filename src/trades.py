"""Moneyball step 5: which player should each team trade for?

For every team, try adding each realistic trade target and measure how many extra
wins it adds over the rest of the season:

  1. The team gives up its lowest-rated rotation player (8+ minutes a game).
  2. The new player keeps the minutes he plays now (max 36).
  3. Extra minutes come out of the team's lowest-rated players first (the bench).
  4. Re-compute team strength and expected wins over the remaining schedule.

Buyers: teams projected at .500 or better. Sellers: teams projected under .500.
Realistic targets: healthy players aged 24+ on seller teams. Rebuilding teams usually
trade veterans, not young cornerstones.

Not modeled yet: salary matching, contract length, and team fit (positions, skills).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import project
from project_current import build

OUT = Path(__file__).resolve().parents[1] / "data" / "processed"
ROTATION_MIN = 8
MAX_MIN = 36
SELLER_MAX_WINS = 41
MIN_TARGET_AGE = 24


def team_rating_sum(players: pd.DataFrame) -> float:
    return float((players["RATING"] * players["EXP_MIN"] / 48).sum())


def add_player(team: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Team roster after swapping its worst rotation player for `target`."""
    rot = team[team["EXP_MIN"] >= ROTATION_MIN]
    out_id = rot.loc[rot["RATING"].idxmin(), "PLAYER_ID"]
    new = pd.concat([team[team["PLAYER_ID"] != out_id],
                     target[["PLAYER_ID", "RATING", "EXP_MIN"]].to_frame().T], ignore_index=True)
    new["EXP_MIN"] = new["EXP_MIN"].astype(float)
    new.loc[new["PLAYER_ID"] == target["PLAYER_ID"], "EXP_MIN"] = min(target["EXP_MIN"], MAX_MIN)
    excess = new["EXP_MIN"].sum() - 240
    for i in new.sort_values("RATING").index:  # trim the bench first
        if excess <= 0:
            break
        if new.at[i, "PLAYER_ID"] == target["PLAYER_ID"]:
            continue
        cut = min(new.at[i, "EXP_MIN"], excess)
        new.at[i, "EXP_MIN"] -= cut
        excess -= cut
    return new, out_id


def extra_wins(team_id, new_sum, old_sum, strength, remaining, home_edge):
    """Change in expected wins over the remaining schedule when strength rises by new-old."""
    delta = new_sum - old_sum
    g = remaining[(remaining["HOME_ID"] == team_id) | (remaining["AWAY_ID"] == team_id)]
    home = g["HOME_ID"] == team_id
    opp = np.where(home, g["AWAY_ID"].map(strength), g["HOME_ID"].map(strength))
    edge = np.where(home, home_edge, -home_edge)
    s = strength[team_id]
    before = project.win_prob(s, opp, edge)
    after = project.win_prob(s + delta, opp, edge)
    return float((after - before).sum())


def main(top_n=3):
    res = build()
    ratings, minutes, proj = res["ratings"], res["minutes"], res["proj"]
    abbr, remaining, home_edge = res["abbr"], res["remaining"], res["home_edge"]
    r = ratings.set_index("PLAYER_ID")
    m = minutes.assign(RATING=minutes["PLAYER_ID"].map(r["RATING"]).fillna(-2.0),
                       AGE=minutes["PLAYER_ID"].map(r["AGE"]),
                       NAME=minutes["PLAYER_ID"].map(r["PLAYER_NAME"]))
    strength = proj["STRENGTH"]

    sellers = proj.index[proj["PROJ_WINS"] < SELLER_MAX_WINS]
    injured = set() if res["injuries"] is None else \
        set(res["injuries"].loc[res["injuries"]["GAMES_OUT"] >= 10, "PLAYER_ID"])
    targets = m[m["TEAM_ID"].isin(sellers) & (m["AGE"] >= MIN_TARGET_AGE) &
                (m["EXP_MIN"] >= 15) & ~m["PLAYER_ID"].isin(injured)]

    rows = []
    buyers = proj.index[proj["PROJ_WINS"] >= SELLER_MAX_WINS]
    for team_id in buyers:
        team = m[m["TEAM_ID"] == team_id]
        old = team_rating_sum(team)
        for _, t in targets[targets["TEAM_ID"] != team_id].iterrows():
            new, out_id = add_player(team, t)
            gain = extra_wins(team_id, team_rating_sum(new), old, strength, remaining, home_edge)
            rows.append({"TEAM": abbr[team_id], "TARGET": t["NAME"], "FROM": abbr[t["TEAM_ID"]],
                         "AGE": int(t["AGE"]), "RATING": t["RATING"], "MPG": t["EXP_MIN"],
                         "GIVE_UP": r["PLAYER_NAME"].get(out_id, str(out_id)), "EXTRA_WINS": gain})
    allr = pd.DataFrame(rows)
    best = (allr.sort_values("EXTRA_WINS", ascending=False).groupby("TEAM").head(top_n)
            .merge(proj[["TEAM", "PROJ_WINS"]], on="TEAM")
            .sort_values(["PROJ_WINS", "EXTRA_WINS"], ascending=[False, False]))
    best.to_csv(OUT / "trade_targets_2026-27.csv", index=False)
    print(f"Sellers ({len(sellers)} teams projected under {SELLER_MAX_WINS} wins):",
          ", ".join(sorted(abbr[sellers])))
    print("\nMost-wanted targets (how many teams have them in their top 3):")
    print(best["TARGET"].value_counts().head(8).to_string())
    print(f"\nTop target for each buyer (extra wins over the ~70 games left):")
    print(best.groupby("TEAM", sort=False).head(1)[["TEAM", "PROJ_WINS", "TARGET", "FROM", "AGE",
                                                     "RATING", "GIVE_UP", "EXTRA_WINS"]]
          .round(1).to_string(index=False))


if __name__ == "__main__":
    main()
