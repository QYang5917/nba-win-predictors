"""Live test: log predictions for today's NBA games BEFORE tip-off, using the frozen models.

Runs several times a day; each run handles games tipping in the next few hours, so the
prediction uses an injury report from shortly before the game. For each such game:
  - Game: P(home win) from player-based team strength (ratings x expected minutes of players
    not ruled out on today's official injury report) + rest days.
  - Props: predicted median points for every available player expected to play 20+ minutes.

Predictions are appended to predictions/games.csv and predictions/props.csv with a UTC
timestamp. A game that's already logged is never overwritten (the first prediction counts).
"""
import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

import build_player_strength as bps
from build_player_points import build_features, load_team_logs
from injuries import current_injuries
from ratings import fit_age_curve, fit_box_weights, load_ages, load_logs, season_ratings

ROOT = Path(__file__).resolve().parents[1]
RAW, MODELS, PRED = ROOT / "data" / "raw", ROOT / "models", ROOT / "predictions"
SEASON = "2026-27"
MODEL_VERSION = "v1 (frozen 2026-11-12)"
OUT_CATEGORIES = {"season", "rehab", "long", "short"}  # Out or Doubtful
PROP_MIN_MINUTES = 20
WINDOW_HOURS = 3.5  # predict only games tipping soon, so the latest injury report is used
STAT_COLS = ["MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB", "AST",
             "TOV", "STL", "BLK", "BLKA", "PF", "PFD", "PTS", "PLUS_MINUS"]


def todays_games(now_utc: dt.datetime) -> pd.DataFrame:
    sch = pd.read_csv(RAW / "schedule_current.csv", dtype={"gameId": str})
    sch = sch[sch["gameId"].str.startswith("002") & (sch["homeTeam_teamId"] > 0)]
    today_et = now_utc.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    sch["TIP_UTC"] = pd.to_datetime(sch["gameDateTimeUTC"], utc=True)
    soon = now_utc + dt.timedelta(hours=WINDOW_HOURS)
    g = sch[(sch["gameDateTimeEst"].str[:10] == today_et) & (sch["TIP_UTC"] > now_utc) & (sch["TIP_UTC"] <= soon)]
    return g.rename(columns={"gameId": "GAME_ID", "homeTeam_teamId": "HOME_ID", "awayTeam_teamId": "AWAY_ID",
                             "homeTeam_teamTricode": "HOME", "awayTeam_teamTricode": "AWAY"})[
        ["GAME_ID", "TIP_UTC", "HOME_ID", "AWAY_ID", "HOME", "AWAY"]].reset_index(drop=True)


def append_log(path: Path, new: pd.DataFrame, key: list[str]) -> int:
    PRED.mkdir(exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, dtype={"GAME_ID": str})
        new = new[~new.set_index(key).index.isin(old.set_index(key).index)]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(path, index=False)
    return len(new)


def main():
    now = dt.datetime.now(dt.timezone.utc)
    games = todays_games(now)
    if games.empty:
        print(f"No games tipping in the next {WINDOW_HOURS} hours.")
        return
    today = now.astimezone(ZoneInfo("America/New_York")).date().isoformat()

    logs = load_logs()
    logs["GAME_ID"] = logs["GAME_ID"].str.zfill(10)
    ages = load_ages()
    history = sorted(s for s in logs["SEASON"].unique() if s != SEASON)
    box = fit_box_weights(logs, history)
    ratings = season_ratings(logs, SEASON, box, fit_age_curve(logs, ages, box, SEASON), ages)
    rating = ratings.set_index("PLAYER_ID")["RATING"]

    rosters = pd.read_csv(RAW / "rosters_current.csv").rename(columns={"TeamID": "TEAM_ID"})
    inj = current_injuries(rosters)
    out_ids = set(inj.loc[inj["CATEGORY"].isin(OUT_CATEGORIES), "PLAYER_ID"])
    questionable = set(inj.loc[inj["CATEGORY"] == "questionable", "PLAYER_ID"])
    stale = sorted(set(inj["TEAM_ABBREVIATION"]) & set(games["HOME"]).union(games["AWAY"])
                   - set(inj.loc[inj["REPORT_DATE"] == today, "TEAM_ABBREVIATION"]))
    if stale:
        print(f"Warning: no injury report from today yet for {stale}; using their latest one.")

    # Expected minutes: each player's latest value this season (after his last game)
    em = bps.expected_minutes(logs)
    cur = em[em["SEASON"] == SEASON].sort_values("GAME_DATE").groupby("PLAYER_ID").last()
    roster = rosters[["PLAYER_ID", "TEAM_ID", "PLAYER", "TEAM_ABBREVIATION"]].drop_duplicates("PLAYER_ID")
    avail = roster[roster["PLAYER_ID"].isin(cur.index) & ~roster["PLAYER_ID"].isin(out_ids)].copy()
    avail["EXP"] = avail["PLAYER_ID"].map(cur["EXP_AFTER"])
    avail["RATING"] = avail["PLAYER_ID"].map(rating).fillna(bps.REPLACEMENT)

    # Rest days since each team's last game
    tl = pd.read_csv(RAW / "team_game_logs_current.csv")
    last_game = pd.to_datetime(tl.groupby("TEAM_ID")["GAME_DATE"].max())
    rest = lambda tid: min(5, (pd.Timestamp(today) - last_game.get(tid, pd.Timestamp("2000-01-01"))).days)

    gm = json.loads((MODELS / "game_model.json").read_text())
    rows, pseudo_players, pseudo_teams = [], [], []
    for g in games.itertuples():
        side = {}
        for tid, abbr, opp, home in [(g.HOME_ID, g.HOME, g.AWAY, True), (g.AWAY_ID, g.AWAY, g.HOME, False)]:
            team = avail[avail["TEAM_ID"] == tid].assign(GAME_ID=g.GAME_ID)
            side[home] = float(bps.strength(team, "EXP").iloc[0]) if len(team) else bps.REPLACEMENT * 5
            matchup = f"{abbr} vs. {opp}" if home else f"{abbr} @ {opp}"
            base = dict(SEASON=SEASON, TEAM_ID=tid, TEAM_ABBREVIATION=abbr, GAME_ID=g.GAME_ID,
                        GAME_DATE=today, MATCHUP=matchup, WL=None, **{c: 0 for c in STAT_COLS})
            pseudo_teams.append(base)
            for p in team.itertuples():
                pseudo_players.append({**base, "PLAYER_ID": p.PLAYER_ID, "PLAYER_NAME": p.PLAYER, "FUTURE": True})
        diff, rest_diff = side[True] - side[False], rest(g.HOME_ID) - rest(g.AWAY_ID)
        z = gm["intercept"] + gm["coef"][0] * diff + gm["coef"][1] * rest_diff
        p_home = 1 / (1 + np.exp(-z))
        rows.append(dict(PREDICTED_AT=now.strftime("%Y-%m-%dT%H:%M:%SZ"), GAME_DATE=today, GAME_ID=g.GAME_ID,
                         TIP_UTC=g.TIP_UTC.strftime("%Y-%m-%dT%H:%M:%SZ"), HOME=g.HOME, AWAY=g.AWAY,
                         HOME_STRENGTH=round(side[True], 2), AWAY_STRENGTH=round(side[False], 2),
                         REST_DIFF=rest_diff, P_HOME=round(float(p_home), 4),
                         PICK=g.HOME if p_home >= 0.5 else g.AWAY, MODEL=MODEL_VERSION))
    game_preds = pd.DataFrame(rows)

    # Props: same feature code as training, with tonight's games added as FUTURE rows
    pl = pd.concat([logs.assign(FUTURE=False), pd.DataFrame(pseudo_players)], ignore_index=True)
    team_logs = pd.concat([load_team_logs(), pd.DataFrame(pseudo_teams)], ignore_index=True)
    feats = build_features(pl, team_logs)
    tonight = feats[feats["FUTURE"]].copy()
    pm = joblib.load(MODELS / "points_model.joblib")
    tonight["PRED_PTS"] = pm["model"].predict(tonight[pm["features"]])
    abbr = roster.set_index("PLAYER_ID")["TEAM_ABBREVIATION"]
    opp_abbr = rosters.drop_duplicates("TEAM_ID").set_index("TEAM_ID")["TEAM_ABBREVIATION"]
    props = tonight[tonight["EXP_MIN"] >= PROP_MIN_MINUTES].assign(
        PREDICTED_AT=now.strftime("%Y-%m-%dT%H:%M:%SZ"), GAME_DATE=today,
        TEAM=lambda x: x["PLAYER_ID"].map(abbr), OPP=lambda x: x["OPP_ID"].map(opp_abbr),
        QUESTIONABLE=lambda x: x["PLAYER_ID"].isin(questionable), MODEL=MODEL_VERSION)
    props = props[["PREDICTED_AT", "GAME_DATE", "GAME_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM", "OPP",
                   "QUESTIONABLE", "EXP_MIN", "SEASON_AVG_PTS", "PRED_PTS", "MISSING_USAGE", "MODEL"]].round(2)

    n_g = append_log(PRED / "games.csv", game_preds, ["GAME_ID"])
    n_p = append_log(PRED / "props.csv", props, ["GAME_ID", "PLAYER_ID"])
    print(f"Logged {n_g} new game predictions and {n_p} new prop predictions for {today}.")
    show = game_preds.assign(WIN_PROB=lambda x: np.where(x.P_HOME >= 0.5, x.P_HOME, 1 - x.P_HOME))
    print(show[["AWAY", "HOME", "P_HOME", "PICK", "WIN_PROB"]].to_string(index=False))
    print(props.sort_values("PRED_PTS", ascending=False).head(10)[
        ["PLAYER_NAME", "TEAM", "OPP", "EXP_MIN", "SEASON_AVG_PTS", "PRED_PTS"]].to_string(index=False))


if __name__ == "__main__":
    main()
