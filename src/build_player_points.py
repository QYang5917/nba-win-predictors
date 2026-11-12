"""Player points props, step 1: pregame features to predict how many points a player scores.

Every feature uses only information from before the game:
  - Expected minutes and scoring rates: season-to-date and recent (last ~10 games, weighted),
    blended with last season early on.
  - Opponent defense and pace, and own team pace (season-to-date, blended with last season).
  - Injury boost: shot volume of teammates who played last game but are out tonight
    (MISSING_USAGE), and of teammates returning tonight (RETURNING_USAGE). In real use,
    tonight's absences come from the injury report before props are posted. Small leak:
    a coach's-decision benching isn't known in advance.

Only rows where the player actually played are kept, because props are void if he doesn't play.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "player_points_features.csv"
SPAN = 10           # recent-form window (games)
PRIOR_MIN = 300     # minutes of last season's rates blended in early on
PRIOR_GAMES = 5     # games of last season's minutes per game blended in early on
TEAM_PRIOR = 10     # games of last season's team stats blended in early on


def next_season(s: str) -> str:
    return f"{int(s[:4]) + 1}-{str(int(s[:4]) + 2)[-2:]}"


def load_team_logs() -> pd.DataFrame:
    df = pd.read_csv(RAW / "team_game_logs.csv", dtype={"GAME_ID": str})
    cur = RAW / "team_game_logs_current.csv"
    if cur.exists():
        df = pd.concat([df, pd.read_csv(cur, dtype={"GAME_ID": str})], ignore_index=True)
    df["GAME_ID"] = df["GAME_ID"].str.zfill(10)
    return df.drop_duplicates(["GAME_ID", "TEAM_ID"])


def team_context(df: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: blended offensive/defensive rating and pace BEFORE the game.
    Rows for games not yet played can be included with zero stats (cumulative sums exclude them)."""
    df = df.copy()
    df["POSS"] = df.FGA + 0.44 * df.FTA - df.OREB + df.TOV
    df["G"] = 1
    opp = df[["GAME_ID", "TEAM_ID", "PTS", "POSS"]].rename(
        columns={"TEAM_ID": "OPP_ID", "PTS": "OPP_PTS", "POSS": "OPP_POSS"})
    df = df.merge(opp, on="GAME_ID")
    df = df[df.TEAM_ID != df.OPP_ID].sort_values(["TEAM_ID", "SEASON", "GAME_DATE"]).copy()
    s = ["PTS", "OPP_PTS", "POSS", "OPP_POSS", "G"]

    def rates(t):
        return pd.DataFrame({"ORTG": 100 * t.PTS / t.POSS, "DRTG": 100 * t.OPP_PTS / t.OPP_POSS,
                             "PACE": (t.POSS + t.OPP_POSS) / 2 / t.G})

    cum = df.groupby(["TEAM_ID", "SEASON"])[s].cumsum() - df[s]
    prev = rates(df.groupby(["TEAM_ID", "SEASON"])[s].sum())
    prev.index = prev.index.set_levels([next_season(x) for x in prev.index.levels[1]], level=1)
    prev = prev.reindex(pd.MultiIndex.from_arrays([df.TEAM_ID, df.SEASON])).set_index(df.index)
    prev = prev.fillna(rates(df[s].sum().to_frame().T).iloc[0])
    g = cum["G"]
    b = (rates(cum).fillna(0).mul(g, axis=0) + prev * TEAM_PRIOR).div(g + TEAM_PRIOR, axis=0)
    b[["GAME_ID", "TEAM_ID", "OPP_ID"]] = df[["GAME_ID", "TEAM_ID", "OPP_ID"]]
    b["GAME_ID"] = b["GAME_ID"].str.zfill(10)
    return b


def build_features(logs: pd.DataFrame, team_logs: pd.DataFrame) -> pd.DataFrame:
    """logs: player game logs. Rows with FUTURE=True are games not yet played (zero stats);
    they get features too, so the same code serves history and tonight's predictions."""
    logs = logs.copy()
    if "FUTURE" not in logs:
        logs["FUTURE"] = False
    logs = logs[(logs["MIN"] > 0) | logs["FUTURE"]].copy()
    logs["GAME_ID"] = logs["GAME_ID"].str.zfill(10)
    logs["USG"] = logs.FGA + 0.44 * logs.FTA + logs.TOV  # possessions used
    d = logs.sort_values(["PLAYER_ID", "SEASON", "GAME_DATE"]).reset_index(drop=True)
    key = ["PLAYER_ID", "SEASON"]
    grp = d.groupby(key)

    # season-to-date totals before this game
    for c in ["PTS", "MIN", "FGA", "FTA", "FG3A", "USG"]:
        d["CUM_" + c] = grp[c].cumsum() - d[c]
    d["G_BEFORE"] = grp.cumcount()
    # recent form (exponentially weighted, before this game) and after this game (for next game)
    for c in ["PTS", "MIN", "USG"]:
        after = grp[c].transform(lambda x: x.ewm(span=SPAN).mean())
        d[c + "_EWM_AFTER"] = after
        d[c + "_EWM"] = after.groupby([d.PLAYER_ID, d.SEASON]).shift()

    # last season's per-minute rates and minutes per game
    tot = logs.groupby(key)[["PTS", "MIN", "FGA", "FTA", "FG3A", "USG"]].sum()
    gp = logs.groupby(key).size()
    prev = tot.div(tot["MIN"], axis=0).drop(columns="MIN").add_prefix("PREV_PM_")
    prev["PREV_MPG"] = tot["MIN"] / gp
    prev.index = prev.index.set_levels([next_season(x) for x in prev.index.levels[1]], level=1)
    d = d.join(prev, on=key)
    league_pm = tot.sum() / tot["MIN"].sum()

    for c in ["PTS", "FGA", "FTA", "FG3A", "USG"]:
        p = d["PREV_PM_" + c].fillna(league_pm[c] * 0.8)  # unknown players: a bit below average
        d[c + "_PM"] = (d["CUM_" + c] + PRIOR_MIN * p) / (d["CUM_MIN"] + PRIOR_MIN)
    mpg_now = (d["MIN_EWM"]).fillna(0)
    has_prev = d["PREV_MPG"].notna()
    g = d["G_BEFORE"]
    d["EXP_MIN"] = np.where(has_prev, (g * mpg_now + PRIOR_GAMES * d["PREV_MPG"].fillna(0)) / (g + PRIOR_GAMES),
                            np.where(g > 0, mpg_now, 12.0))
    d["EXP_PTS"] = d["EXP_MIN"] * d["PTS_PM"]  # simple "rate x minutes" baseline
    d["SEASON_AVG_PTS"] = (d["CUM_PTS"] / g).where(g > 0)
    d["HOME"] = d["MATCHUP"].str.contains("vs.").astype(int)
    d["GAME_DATE"] = pd.to_datetime(d["GAME_DATE"])
    d["REST"] = grp["GAME_DATE"].transform(lambda x: pd.to_datetime(x).diff().dt.days).clip(upper=5).fillna(5)

    # injury boost: teammates who played the team's previous game but not tonight, and vice versa
    tg = d[["GAME_ID", "TEAM_ID", "SEASON", "GAME_DATE"]].drop_duplicates(["GAME_ID", "TEAM_ID"])
    tg = tg.sort_values(["TEAM_ID", "SEASON", "GAME_DATE"])
    tg["PREV_GAME_ID"] = tg.groupby(["TEAM_ID", "SEASON"])["GAME_ID"].shift()
    d = d.merge(tg[["GAME_ID", "TEAM_ID", "PREV_GAME_ID"]], on=["GAME_ID", "TEAM_ID"])
    usg_after = d.set_index(["GAME_ID", "TEAM_ID", "PLAYER_ID"])["USG_EWM_AFTER"]
    usg_before = d.set_index(["GAME_ID", "TEAM_ID", "PLAYER_ID"])["USG_EWM"].fillna(0)
    prev_players = d[["PREV_GAME_ID", "GAME_ID", "TEAM_ID"]].drop_duplicates().dropna()
    last = prev_players.merge(
        d[["GAME_ID", "TEAM_ID", "PLAYER_ID"]].rename(columns={"GAME_ID": "PREV_GAME_ID"}),
        on=["PREV_GAME_ID", "TEAM_ID"])
    last["USG"] = usg_after.reindex(pd.MultiIndex.from_arrays(
        [last.PREV_GAME_ID, last.TEAM_ID, last.PLAYER_ID])).to_numpy()
    tonight = set(zip(d.GAME_ID, d.TEAM_ID, d.PLAYER_ID))
    last["OUT"] = [(g_, t, p) not in tonight for g_, t, p in zip(last.GAME_ID, last.TEAM_ID, last.PLAYER_ID)]
    missing = last[last.OUT].groupby(["GAME_ID", "TEAM_ID"])["USG"].sum().rename("MISSING_USAGE")
    was_there = set(zip(last.GAME_ID, last.TEAM_ID, last.PLAYER_ID))
    d["RETURNING"] = [(g_, t, p) not in was_there and pg is not None and isinstance(pg, str)
                      for g_, t, p, pg in zip(d.GAME_ID, d.TEAM_ID, d.PLAYER_ID, d.PREV_GAME_ID)]
    ret = (usg_before.to_numpy() * d["RETURNING"]).groupby([d.GAME_ID, d.TEAM_ID]).sum().rename("RETURNING_USAGE")
    d = d.join(missing, on=["GAME_ID", "TEAM_ID"]).join(ret, on=["GAME_ID", "TEAM_ID"])
    d[["MISSING_USAGE", "RETURNING_USAGE"]] = d[["MISSING_USAGE", "RETURNING_USAGE"]].fillna(0)
    # the player's own share of team volume (stars absorb more of a missing teammate's shots)
    d["TEAM_USG_EXP"] = (d["USG_PM"] * d["EXP_MIN"]).groupby([d.GAME_ID, d.TEAM_ID]).transform("sum")
    d["USG_SHARE"] = d["USG_PM"] * d["EXP_MIN"] / d["TEAM_USG_EXP"]

    ctx = team_context(team_logs)
    d = d.merge(ctx[["GAME_ID", "TEAM_ID", "OPP_ID", "PACE", "ORTG"]].rename(
        columns={"PACE": "TEAM_PACE", "ORTG": "TEAM_ORTG"}), on=["GAME_ID", "TEAM_ID"])
    opp = ctx[["GAME_ID", "TEAM_ID", "PACE", "DRTG"]].rename(
        columns={"TEAM_ID": "OPP_ID", "PACE": "OPP_PACE", "DRTG": "OPP_DRTG"})
    d = d.merge(opp, on=["GAME_ID", "OPP_ID"])

    cols = ["GAME_ID", "SEASON", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "FUTURE", "PTS",
            "EXP_MIN", "EXP_PTS", "SEASON_AVG_PTS", "PTS_EWM", "MIN_EWM", "PTS_PM", "FGA_PM", "FTA_PM",
            "FG3A_PM", "USG_PM", "USG_SHARE", "G_BEFORE", "HOME", "REST", "MISSING_USAGE",
            "RETURNING_USAGE", "TEAM_PACE", "TEAM_ORTG", "OPP_PACE", "OPP_DRTG", "OPP_ID"]
    d = d[cols].copy()
    d["MISS_X_SHARE"] = d["MISSING_USAGE"] * d["USG_SHARE"]
    d["RET_X_SHARE"] = d["RETURNING_USAGE"] * d["USG_SHARE"]
    return d


FEATURES = ["EXP_MIN", "EXP_PTS", "PTS_EWM", "MIN_EWM", "PTS_PM", "FGA_PM", "FTA_PM", "FG3A_PM", "USG_PM",
            "USG_SHARE", "G_BEFORE", "HOME", "REST", "MISSING_USAGE", "RETURNING_USAGE", "TEAM_PACE",
            "TEAM_ORTG", "OPP_PACE", "OPP_DRTG", "MISS_X_SHARE", "RET_X_SHARE"]


def main():
    logs = pd.read_csv(RAW / "player_game_logs.csv.gz", dtype={"GAME_ID": str})
    d = build_features(logs, load_team_logs())
    d.to_csv(OUT, index=False)
    print(f"Saved {len(d)} player-games to {OUT.name}")


if __name__ == "__main__":
    main()
