"""Moneyball step 4: project 2026-27 final wins for all 30 teams, as of today.

Inputs: player ratings (this season + last two, age-adjusted), current rosters,
the official injury report, and the remaining schedule.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import project
from injuries import current_injuries
from project import expected_minutes, expected_wins, team_strength
from ratings import fit_age_curve, fit_box_weights, load_ages, load_logs, season_ratings

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
SEASON = "2026-27"
EAST = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL",
        "NYK", "ORL", "PHI", "TOR", "WAS"}


def load_current():
    rosters = pd.read_csv(RAW / "rosters_current.csv").rename(columns={"TeamID": "TEAM_ID"})
    sch = pd.read_csv(RAW / "schedule_current.csv", dtype={"gameId": str})
    sch = sch[sch["gameId"].str.startswith("002")]  # regular season only
    games = pd.read_csv(RAW / "team_game_logs_current.csv", dtype={"GAME_ID": str})
    wins = games[games["WL"] == "W"].groupby("TEAM_ID").size()
    played = games.groupby("TEAM_ID").size()
    done = set(games["GAME_ID"])
    remaining = sch[~sch["gameId"].isin(done)].rename(
        columns={"homeTeam_teamId": "HOME_ID", "awayTeam_teamId": "AWAY_ID"})[["HOME_ID", "AWAY_ID"]]
    remaining = remaining[(remaining["HOME_ID"] > 0) & (remaining["AWAY_ID"] > 0)]  # drop TBD slots
    abbr = rosters.drop_duplicates("TEAM_ID").set_index("TEAM_ID")["TEAM_ABBREVIATION"]
    return rosters, remaining, wins.reindex(abbr.index, fill_value=0), played.reindex(abbr.index, fill_value=0), abbr


def project_wins(strength, remaining, wins, played, home_edge):
    proj = expected_wins(strength, remaining, wins, home_edge)
    # NBA Cup knockout games aren't scheduled yet: treat as neutral games vs. an average team.
    proj["UNSCHEDULED"] = (82 - played - proj["GAMES_LEFT"]).clip(lower=0)
    proj["PROJ_WINS"] += proj["UNSCHEDULED"] * norm.cdf(proj["STRENGTH"] / project.SIGMA)
    return proj


def build(use_injuries=True):
    logs = load_logs()
    ages = load_ages()
    history = sorted(s for s in logs["SEASON"].unique() if s != SEASON)
    box_model = fit_box_weights(logs, history)
    curve = fit_age_curve(logs, ages, box_model, SEASON)
    ratings = season_ratings(logs, SEASON, box_model, curve, ages)

    rosters, remaining, wins, played, abbr = load_current()
    left = pd.concat([remaining["HOME_ID"], remaining["AWAY_ID"]]).value_counts()
    games_left = (82 - played).where(lambda s: s > 0, left)
    injuries = current_injuries(rosters) if use_injuries else None
    minutes = expected_minutes(logs, SEASON, "2099-01-01", rosters[["PLAYER_ID", "TEAM_ID"]],
                               injuries, games_left)
    strength = team_strength(minutes, ratings)
    proj = project_wins(strength, remaining, wins, played, ratings.attrs["home_edge"])
    proj["TEAM"] = abbr
    proj["CONF"] = np.where(proj["TEAM"].isin(EAST), "East", "West")
    proj["RECORD"] = wins.astype(int).astype(str) + "-" + (played - wins).astype(int).astype(str)
    return dict(ratings=ratings, minutes=minutes, injuries=injuries, proj=proj,
                remaining=remaining, wins=wins, played=played, abbr=abbr,
                home_edge=ratings.attrs["home_edge"])


if __name__ == "__main__":
    res = build()
    no_inj = build(use_injuries=False)["proj"]["PROJ_WINS"]
    p = res["proj"].assign(INJURY_EFFECT=lambda d: d["PROJ_WINS"] - no_inj)
    p = p.sort_values("PROJ_WINS", ascending=False)
    p.to_csv(OUT / "projections_2026-27.csv")
    res["ratings"].to_csv(OUT / "player_ratings_current.csv", index=False)
    for conf in ["East", "West"]:
        print(f"\n{conf}")
        print(p[p["CONF"] == conf][["TEAM", "RECORD", "STRENGTH", "PROJ_WINS", "INJURY_EFFECT"]]
              .round(1).to_string(index=False))
