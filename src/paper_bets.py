"""Paper bets: pretend $1 bets on picks the frozen models rate as positive value. No real money.

A pick has positive value when (model's chance of winning) x (payout) is more than $1.
Two markets are checked, always at the best price among the recorded sportsbooks:
  - Moneyline: the game model's win chance for each team.
  - Player points: the points model's median, turned into an over/under chance with
    models/points_distribution.json (see build_points_distribution.py).

Rules, fixed on 2026-11-16 before any bet was logged:
  - A bet is logged only before tip-off, from a prediction and a line that were both saved before tip-off.
  - One bet at most per game (moneyline) and per player (points). Logged bets are never changed.
  - Every pick with positive value is logged. The MAIN rule that gets judged is value above 5%.
  - A points bet is void (stake returned) if the player doesn't play.

  predictions/paper_bets.csv   the bets and, once games finish, their results
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_points_distribution import prob_over

ROOT = Path(__file__).resolve().parents[1]
MODELS, PRED = ROOT / "models", ROOT / "predictions"
BETS = PRED / "paper_bets.csv"
MAIN_EDGE = 0.05
COLS = ["LOGGED_AT", "GAME_DATE", "GAME_ID", "TIP_UTC", "MARKET", "PLAYER_ID", "PLAYER_NAME", "PICK", "LINE",
        "BOOK", "PRICE", "MODEL_PROB", "EV", "RESULT", "PROFIT"]


def read(name: str) -> pd.DataFrame:
    path = PRED / name
    return pd.read_csv(path, dtype={"GAME_ID": str}) if path.exists() else pd.DataFrame()


def before_tip(made_at, tip) -> pd.Series:
    return pd.to_datetime(made_at, utc=True, errors="coerce") < pd.to_datetime(tip, utc=True, errors="coerce")


def moneyline_bets(games: pd.DataFrame, lines: pd.DataFrame) -> list[dict]:
    """Best price for each team across books; bet the side with the higher positive value."""
    if games.empty or lines.empty:
        return []
    ml = lines[(lines["MARKET"] == "h2h") & before_tip(lines["PULLED_AT"], lines["TIP_UTC"])]
    best = ml.loc[ml.groupby(["GAME_ID", "SIDE"])["PRICE"].idxmax()].set_index(["GAME_ID", "SIDE"])
    rows = []
    for g in games.itertuples():
        options = []
        for side, team, p in [("HOME", g.HOME, g.P_HOME), ("AWAY", g.AWAY, 1 - g.P_HOME)]:
            if (g.GAME_ID, side) in best.index:
                b = best.loc[(g.GAME_ID, side)]
                options.append(dict(PICK=team, BOOK=b["BOOK"], PRICE=b["PRICE"], MODEL_PROB=round(p, 4),
                                    EV=round(p * b["PRICE"] - 1, 4)))
        if options and max(o["EV"] for o in options) > 0:
            rows.append(dict(GAME_DATE=g.GAME_DATE, GAME_ID=g.GAME_ID, TIP_UTC=g.TIP_UTC, MARKET="moneyline",
                             **max(options, key=lambda o: o["EV"])))
    return rows


def points_bets(props: pd.DataFrame, lines: pd.DataFrame, tips: dict, bins: list[dict]) -> list[dict]:
    """For each player, the single best over or under among all books, if its value is positive."""
    if props.empty or lines.empty:
        return []
    ln = lines.dropna(subset=["PLAYER_ID", "LINE"]).copy()
    ln = ln[before_tip(ln["PULLED_AT"], ln["GAME_ID"].map(tips))]
    pred = props.set_index(["GAME_ID", "PLAYER_ID"])
    rows = []
    for (gid, pid), grp in ln.groupby(["GAME_ID", "PLAYER_ID"]):
        if (gid, pid) not in pred.index:
            continue
        p = pred.loc[(gid, pid)]
        options = []
        for b in grp.itertuples():
            over = prob_over(bins, p["PRED_PTS"], b.LINE)
            for pick, prob, price in [("OVER", over, b.OVER_PRICE), ("UNDER", 1 - over, b.UNDER_PRICE)]:
                if pd.notna(price):
                    options.append(dict(PICK=pick, LINE=b.LINE, BOOK=b.BOOK, PRICE=price,
                                        MODEL_PROB=round(prob, 4), EV=round(prob * price - 1, 4)))
        if options and max(o["EV"] for o in options) > 0:
            rows.append(dict(GAME_DATE=p["GAME_DATE"], GAME_ID=gid, TIP_UTC=tips[gid], MARKET="points",
                             PLAYER_ID=int(pid), PLAYER_NAME=p["PLAYER_NAME"], **max(options, key=lambda o: o["EV"])))
    return rows


def log_new(now: dt.datetime) -> int:
    games, props = read("games.csv"), read("props.csv")
    if games.empty:
        return 0
    games = games[before_tip(games["PREDICTED_AT"], games["TIP_UTC"])]
    open_ids = set(games.loc[pd.to_datetime(games["TIP_UTC"], utc=True) > now, "GAME_ID"])  # not tipped yet
    games = games[games["GAME_ID"].isin(open_ids)]
    tips = dict(zip(games["GAME_ID"], games["TIP_UTC"]))
    props = props[props["GAME_ID"].isin(open_ids)] if len(props) else props
    bins = json.loads((MODELS / "points_distribution.json").read_text())["bins"]
    rows = moneyline_bets(games, read("lines_games.csv")) + points_bets(props, read("lines_props.csv"), tips, bins)
    new = pd.DataFrame(rows).reindex(columns=COLS)
    new["LOGGED_AT"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    old = read("paper_bets.csv")
    if len(old) and len(new):  # one bet per game (moneyline) or per player (points); never replaced
        key = lambda d: d["GAME_ID"] + "|" + d["MARKET"] + "|" + d["PLAYER_ID"].fillna(0).astype(int).astype(str)
        new = new[~key(new).isin(key(old))]
    if len(new):
        pd.concat([old, new], ignore_index=True)[COLS].to_csv(BETS, index=False)
    return len(new)


def grade() -> pd.DataFrame:
    bets = read("paper_bets.csv")
    if bets.empty:
        return bets
    games, props = read("games.csv"), read("props.csv")
    winner = games.dropna(subset=["HOME_WIN"]).assign(
        W=lambda x: np.where(x["HOME_WIN"].astype(float) == 1, x["HOME"], x["AWAY"])).set_index("GAME_ID")["W"]
    finished = set(winner.index)
    actual = props.set_index(["GAME_ID", "PLAYER_ID"])[["ACTUAL_PTS", "ACTUAL_MIN"]] if len(props) else None
    played = set(props.loc[props["ACTUAL_MIN"] > 0, "GAME_ID"]) if len(props) else set()
    bets["RESULT"] = bets["RESULT"].astype(object)
    for i, b in bets.iterrows():
        if pd.notna(b["RESULT"]) or b["GAME_ID"] not in finished:
            continue
        if b["MARKET"] == "moneyline":
            won = winner[b["GAME_ID"]] == b["PICK"]
        else:
            a = actual.loc[(b["GAME_ID"], b["PLAYER_ID"])]
            if b["GAME_ID"] not in played:  # player results for this game aren't in yet; try next run
                continue
            if not a["ACTUAL_MIN"] > 0:  # didn't play: stake returned
                bets.loc[i, ["RESULT", "PROFIT"]] = ["void", 0.0]
                continue
            won = a["ACTUAL_PTS"] > b["LINE"] if b["PICK"] == "OVER" else a["ACTUAL_PTS"] < b["LINE"]
        bets.loc[i, ["RESULT", "PROFIT"]] = ["won", round(b["PRICE"] - 1, 4)] if won else ["lost", -1.0]
    bets[COLS].to_csv(BETS, index=False)
    return bets


def scorecard_section(bets: pd.DataFrame, n_lines: int) -> list[str]:
    lines = ["", "## Paper bets (no real money)", "",
             "A pretend $1 bet is logged before tip-off whenever the frozen model's chance times the best "
             f"recorded price shows positive value. The main rule, fixed in advance, is value above {MAIN_EDGE:.0%}.", ""]
    done = bets[bets["RESULT"].isin(["won", "lost"])] if len(bets) else bets
    if not len(done):
        waiting = len(bets) if len(bets) else 0
        return lines + [f"No settled bets yet ({waiting} waiting for results; betting lines recorded for {n_lines} games)."]
    lines += ["| Market | Rule | Bets | Won | Return per $1 | Could be luck? |", "|---|---|---|---|---|---|"]
    for market in ["moneyline", "points"]:
        for label, edge in [("any positive value", 0), (f"**value above {MAIN_EDGE:.0%} (main rule)**", MAIN_EDGE),
                            ("value above 10%", 0.10)]:
            s = done[(done["MARKET"] == market) & (done["EV"] > edge)]
            if not len(s):
                lines.append(f"| {market} | {label} | 0 | – | – | – |")
                continue
            roi, se = s["PROFIT"].mean(), s["PROFIT"].std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan
            if len(s) < 30 or not np.isfinite(se):
                verdict = "too few bets to say"
            elif abs(roi) < 2 * se:
                verdict = f"yes (±{2 * se:.0%})"
            else:
                verdict = f"unlikely (±{2 * se:.0%})"
            lines.append(f"| {market} | {label} | {len(s)} | {(s['RESULT'] == 'won').mean():.1%} | {roi:+.1%} | {verdict} |")
    void = (bets["RESULT"] == "void").sum()
    lines += ["", f"Waiting for results: {bets['RESULT'].isna().sum()}. Void (player didn't play): {void}. "
              "Points lines can be recorded a few hours after the prediction, so some bets use an older prediction "
              "against a newer line."]
    return lines


def main():
    now = dt.datetime.now(dt.timezone.utc)
    grade()
    n = log_new(now)
    bets = read("paper_bets.csv")
    print(f"Paper bets: {n} new, {len(bets)} in total.")
    card = PRED / "SCORECARD.md"
    if card.exists():  # grade_predictions.py rewrites the scorecard each run; add this section to the end
        text = card.read_text().split("\n## Paper bets")[0].rstrip("\n")
        lg = read("lines_games.csv")
        section = scorecard_section(bets, lg["GAME_ID"].nunique() if len(lg) else 0)
        card.write_text(text + "\n" + "\n".join(section) + "\n")


if __name__ == "__main__":
    main()
