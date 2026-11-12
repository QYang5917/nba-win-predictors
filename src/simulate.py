"""Moneyball step 6: championship odds by simulating the rest of the season 20,000 times.

Each simulated season:
  1. Draw every team's "true" strength = our estimate + random error (sd 3.3 points,
     measured from how far estimates were off in the backtest).
  2. Play every remaining regular-season game (win chance from the strength gap + home court).
  3. Seed each conference: 1-6 go straight to the playoffs, 7-10 go to the play-in.
  4. Play-in: 7 v 8 (winner = 7 seed), loser v winner of 9 v 10 (winner = 8 seed).
  5. Best-of-seven series (2-2-1-1-1 home court to the better record) through the Finals.
Count how often each team wins the title.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import project
from project_current import build

OUT = Path(__file__).resolve().parents[1] / "data" / "processed"
N_SIMS = 20_000
STRENGTH_SD = 3.3
SEED = 2026


def series_win_prob(p_home: float, p_away: float) -> float:
    """P(team with home court wins a best-of-7), 2-2-1-1-1 format."""
    home_games = [1, 1, 0, 0, 1, 0, 1]
    probs = {(0, 0): 1.0}
    win = 0.0
    for g in range(7):
        nxt = {}
        p = p_home if home_games[g] else p_away
        for (w, l), pr in probs.items():
            for (dw, dl, q) in [(1, 0, p), (0, 1, 1 - p)]:
                s = (w + dw, l + dl)
                if s[0] == 4:
                    win += pr * q
                elif s[1] < 4:
                    nxt[s] = nxt.get(s, 0) + pr * q
        probs = nxt
    return win


def main():
    res = build()
    proj, remaining, edge = res["proj"], res["remaining"], res["home_edge"]
    teams = proj.index.to_numpy()
    idx = {t: i for i, t in enumerate(teams)}
    abbr = proj["TEAM"].to_numpy()
    conf = proj["CONF"].to_numpy()
    base = proj["STRENGTH"].to_numpy()
    rng = np.random.default_rng(SEED)

    # --- regular season (vectorized over simulations) ---
    true_s = base + rng.normal(0, STRENGTH_SD, size=(N_SIMS, len(teams)))
    h = remaining["HOME_ID"].map(idx).to_numpy()
    a = remaining["AWAY_ID"].map(idx).to_numpy()
    p_home = norm.cdf((true_s[:, h] - true_s[:, a] + edge) / project.SIGMA)
    home_won = rng.random(p_home.shape) < p_home
    wins = np.tile(proj["WINS_SO_FAR"].to_numpy(float), (N_SIMS, 1))
    np.add.at(wins.T, h, home_won.T)
    np.add.at(wins.T, a, ~home_won.T)
    unsched = proj["UNSCHEDULED"].to_numpy().astype(int)
    wins += rng.binomial(unsched, norm.cdf(true_s / project.SIGMA))
    order_key = wins + rng.random(wins.shape) * 0.01  # random tie-breaks

    counts = {k: np.zeros(len(teams)) for k in ["PLAYOFFS", "CONF_FINALS", "FINALS", "CHAMPION"]}

    def game(s, i, j):  # single game, i at home
        return i if rng.random() < norm.cdf((s[i] - s[j] + edge) / project.SIGMA) else j

    def series(s, w, i, j):  # better record gets home court
        if w[j] > w[i]:
            i, j = j, i
        ph = norm.cdf((s[i] - s[j] + edge) / project.SIGMA)
        pa = norm.cdf((s[i] - s[j] - edge) / project.SIGMA)
        return i if rng.random() < series_win_prob(ph, pa) else j

    for k in range(N_SIMS):
        s, w = true_s[k], order_key[k]
        champs = []
        for c in ["East", "West"]:
            members = np.where(conf == c)[0]
            seeds = members[np.argsort(-w[members])]
            s7 = game(s, seeds[6], seeds[7])
            loser78 = seeds[7] if s7 == seeds[6] else seeds[6]
            s8 = game(s, loser78, game(s, seeds[8], seeds[9]))
            bracket = [seeds[0], s8, seeds[3], seeds[4], seeds[2], seeds[5], seeds[1], s7]
            counts["PLAYOFFS"][bracket] += 1
            r1 = [series(s, w, bracket[i], bracket[i + 1]) for i in range(0, 8, 2)]
            r2 = [series(s, w, r1[0], r1[1]), series(s, w, r1[2], r1[3])]
            counts["CONF_FINALS"][r2] += 1
            champ = series(s, w, r2[0], r2[1])
            counts["FINALS"][champ] += 1
            champs.append(champ)
        counts["CHAMPION"][series(s, w, champs[0], champs[1])] += 1

    out = pd.DataFrame({"TEAM": abbr, "CONF": conf, "PROJ_WINS": proj["PROJ_WINS"].to_numpy(),
                        "WINS_P10": np.percentile(wins, 10, axis=0),
                        "WINS_P90": np.percentile(wins, 90, axis=0),
                        **{k: v / N_SIMS for k, v in counts.items()}})
    out = out.sort_values("CHAMPION", ascending=False)
    out.to_csv(OUT / "championship_odds_2026-27.csv", index=False)
    show = out.copy()
    for col in counts:
        show[col] = (show[col] * 100).map(lambda v: f"{v:.1f}%")
    show["WIN_RANGE"] = show["WINS_P10"].round().astype(int).astype(str) + "-" + \
        show["WINS_P90"].round().astype(int).astype(str)
    print(f"{N_SIMS:,} simulated seasons. Title odds sum to {out['CHAMPION'].sum():.0%}.")
    print(show[["TEAM", "CONF", "PROJ_WINS", "WIN_RANGE", "PLAYOFFS", "CONF_FINALS", "FINALS",
                "CHAMPION"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
