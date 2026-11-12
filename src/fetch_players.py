"""Moneyball step 1: player game logs, current rosters, and this season's schedule.

- Player game logs (one row per player per game) for 2015-16 through today.
  Game-level rows tell us which team a player was on for each game, so
  mid-season trades are handled correctly.
- Current rosters for all 30 teams (includes injured players who haven't played yet).
- The full 2026-27 regular-season schedule, so we can simulate the games left to play.
"""
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (commonteamroster, leaguegamefinder,
                                     playergamelogs, scheduleleaguev2)
from nba_api.stats.static import teams

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
CURRENT = "2026-27"
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2015, 2027)]


def with_retry(fn, label, retries=4):
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            print(f"  {label}: attempt {attempt} failed ({exc}); retrying...")
            time.sleep(5 * attempt)
    raise RuntimeError(f"Could not fetch {label}")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    # 1. Player game logs
    frames = []
    for s in SEASONS:
        df = with_retry(lambda: playergamelogs.PlayerGameLogs(
            season_nullable=s, season_type_nullable="Regular Season",
            league_id_nullable="00", timeout=90).get_data_frames()[0], s)
        df["SEASON"] = s
        print(f"{s}: {len(df):>6} player-game rows")
        frames.append(df)
        time.sleep(1.5)
    keep = ["SEASON", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID",
            "GAME_DATE", "MATCHUP", "WL", "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
            "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "BLKA", "PF", "PFD", "PTS", "PLUS_MINUS"]
    logs = pd.concat(frames, ignore_index=True)[keep]
    logs["GAME_DATE"] = logs["GAME_DATE"].str[:10]
    # Compressed and trimmed: the full file is >100 MB, over GitHub's per-file limit.
    logs.to_csv(RAW / "player_game_logs.csv.gz", index=False, compression="gzip")

    # 2. This season's team game logs (the original fetch covers 2016-17 to 2025-26)
    cur = with_retry(lambda: leaguegamefinder.LeagueGameFinder(
        player_or_team_abbreviation="T", season_nullable=CURRENT,
        season_type_nullable="Regular Season", league_id_nullable="00",
        timeout=60).get_data_frames()[0], "current team logs")
    cur["SEASON"] = CURRENT
    cur.to_csv(RAW / "team_game_logs_current.csv", index=False)
    print(f"{CURRENT} team games so far: {len(cur) // 2}")

    # 3. Current rosters
    rosters = []
    for t in teams.get_teams():
        r = with_retry(lambda: commonteamroster.CommonTeamRoster(
            team_id=t["id"], season=CURRENT, timeout=60).get_data_frames()[0], t["abbreviation"])
        r["TEAM_ABBREVIATION"] = t["abbreviation"]
        rosters.append(r)
        time.sleep(0.8)
    pd.concat(rosters, ignore_index=True).to_csv(RAW / "rosters_current.csv", index=False)
    print(f"Rosters: {sum(len(r) for r in rosters)} players")

    # 4. Schedule
    sch = with_retry(lambda: scheduleleaguev2.ScheduleLeagueV2(
        season=CURRENT, timeout=60).get_data_frames()[0], "schedule")
    sch.to_csv(RAW / "schedule_current.csv", index=False)
    print(f"Schedule: {len(sch)} rows")


if __name__ == "__main__":
    main()
