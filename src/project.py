"""Moneyball step 3-4: turn player ratings into team strength and projected wins.

team strength (points per game vs. average) = sum over the roster of
    RATING * expected minutes per team game / 48

Expected minutes blend this season's minutes per game with last season's, times how
often the player has been available, scaled so every team plays 240 minutes a game.

Each remaining game: home margin ~ Normal(home strength - away strength + home edge, SIGMA),
so P(home win) = Phi((home - away + edge) / SIGMA).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from ratings import prev_season

SIGMA = 13.0           # spread of single-game margins around expectation (NBA ~12-13 pts)
PRIOR_GAMES = 10       # how many games of last season's minutes to blend with this season's


def season_start(logs: pd.DataFrame, season: str) -> pd.Timestamp:
    return pd.to_datetime(logs.loc[logs["SEASON"] == season, "GAME_DATE"]).min()


def expected_minutes(logs, season, cutoff, roster=None, injuries=None,
                     games_left=None) -> pd.DataFrame:
    """Expected minutes per team game for every player on every team at `cutoff`.

    roster: optional DataFrame (PLAYER_ID, TEAM_ID) of current rosters, which adds
    players who are on a team but haven't played yet this season (e.g. injured).
    injuries: optional DataFrame (PLAYER_ID, GAMES_OUT) from the official injury report.
    games_left: Series TEAM_ID -> games remaining (needed with `injuries`).
    """
    cur = logs[(logs["SEASON"] == season) & (logs["GAME_DATE"] < cutoff)]
    prev = logs[logs["SEASON"] == prev_season(season)]

    team_games = cur.groupby("TEAM_ID")["GAME_ID"].nunique()
    last_team = cur.sort_values("GAME_DATE").groupby("PLAYER_ID")["TEAM_ID"].last()
    c = cur.groupby("PLAYER_ID").agg(GP=("GAME_ID", "nunique"), MIN=("MIN", "sum"))
    p = prev.groupby("PLAYER_ID").agg(GP_P=("GAME_ID", "nunique"), MIN_P=("MIN", "sum"))
    prev_team_games = prev.groupby("TEAM_ID")["GAME_ID"].nunique().mean()

    team = last_team.rename("TEAM_ID").to_frame()
    if roster is not None:  # current roster is the source of truth for who is where
        team = roster.set_index("PLAYER_ID")[["TEAM_ID"]]
    df = team.join(c).join(p).fillna(0)

    mpg_cur = (df["MIN"] / df["GP"]).where(df["GP"] > 0, 0)
    mpg_prev = (df["MIN_P"] / df["GP_P"]).where(df["GP_P"] > 0, np.nan)
    w = df["GP"] / (df["GP"] + PRIOR_GAMES)
    df["MPG"] = np.where(mpg_prev.isna(), mpg_cur, w * mpg_cur + (1 - w) * mpg_prev.fillna(0))

    tg = df["TEAM_ID"].map(team_games).fillna(0)
    avail_cur = (df["GP"] / tg).clip(upper=1).where(tg > 0, np.nan)
    avail_prev = (df["GP_P"] / prev_team_games).clip(upper=1)
    tw = tg / (tg + PRIOR_GAMES)
    avail = np.where(avail_cur.isna(), avail_prev, tw * avail_cur.fillna(0) + (1 - tw) * avail_prev)
    not_yet = (df["GP"] == 0) & (tg >= 5)
    if injuries is None:
        # No injury info: a rostered player who hasn't played is assumed out ~half the season.
        avail = np.where(not_yet, 0.5 * avail_prev, avail)
    else:
        out = df.index.to_series().map(injuries.set_index("PLAYER_ID")["GAMES_OUT"])
        left = df["TEAM_ID"].map(games_left).fillna(70)
        healthy_share = (1 - out / left).clip(lower=0).fillna(1.0)
        # Injured: availability of a healthy version of him, times the share of games he'll be back for.
        base = np.where(not_yet, avail_prev, avail)
        avail = np.where(out.notna(), base * healthy_share, avail)
        # Healthy but hasn't played all season: he's out of the rotation.
        avail = np.where(not_yet & out.isna(), 0.1 * avail_prev, avail)
        df["INJURY_SHARE_OUT"] = 1 - healthy_share
    df["EXP_MIN"] = df["MPG"] * avail

    # Scale each team to 240 minutes per game.
    df["EXP_MIN"] *= 240 / df.groupby("TEAM_ID")["EXP_MIN"].transform("sum")
    return df.reset_index().rename(columns={"index": "PLAYER_ID"})


def team_strength(minutes: pd.DataFrame, ratings: pd.DataFrame, replacement=-2.0) -> pd.Series:
    r = ratings.set_index("PLAYER_ID")["RATING"]
    m = minutes.assign(RATING=minutes["PLAYER_ID"].map(r).fillna(replacement))
    s = (m["RATING"] * m["EXP_MIN"] / 48).groupby(m["TEAM_ID"]).sum()
    return s - s.mean()  # league average = 0


def win_prob(home_strength, away_strength, home_edge):
    return norm.cdf((home_strength - away_strength + home_edge) / SIGMA)


def expected_wins(strength: pd.Series, remaining: pd.DataFrame, wins_so_far: pd.Series,
                  home_edge: float) -> pd.DataFrame:
    """remaining: DataFrame with HOME_ID, AWAY_ID for games not yet played."""
    p = win_prob(remaining["HOME_ID"].map(strength).to_numpy(),
                 remaining["AWAY_ID"].map(strength).to_numpy(), home_edge)
    exp = pd.concat([pd.Series(p, index=remaining["HOME_ID"]),
                     pd.Series(1 - p, index=remaining["AWAY_ID"])]).groupby(level=0).sum()
    left = pd.concat([remaining["HOME_ID"], remaining["AWAY_ID"]]).value_counts()
    out = pd.DataFrame({"STRENGTH": strength, "WINS_SO_FAR": wins_so_far,
                        "EXP_REMAINING": exp, "GAMES_LEFT": left}).fillna(0)
    out["PROJ_WINS"] = out["WINS_SO_FAR"] + out["EXP_REMAINING"]
    return out
