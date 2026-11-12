"""Daily refresh of THIS season only (fast): player logs, team logs, rosters, schedule, injury reports.

Current-season player logs go to a separate small file so the big historical file never changes.
"""
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import commonteamroster, leaguegamefinder, playergamelogs, scheduleleaguev2
from nba_api.stats.static import teams

from fetch_players import with_retry

SRC = Path(__file__).resolve().parent
RAW = SRC.parent / "data" / "raw"
CURRENT = "2026-27"
KEEP = ["SEASON", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID",
        "GAME_DATE", "MATCHUP", "WL", "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
        "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "BLKA", "PF", "PFD", "PTS", "PLUS_MINUS"]


def main():
    logs = with_retry(lambda: playergamelogs.PlayerGameLogs(
        season_nullable=CURRENT, season_type_nullable="Regular Season",
        league_id_nullable="00", timeout=90).get_data_frames()[0], "player logs")
    logs["SEASON"] = CURRENT
    logs["GAME_DATE"] = logs["GAME_DATE"].str[:10]
    logs[KEEP].to_csv(RAW / "player_game_logs_current.csv.gz", index=False, compression="gzip")
    print(f"Player logs {CURRENT}: {len(logs)} rows, through {logs['GAME_DATE'].max()}")

    tl = with_retry(lambda: leaguegamefinder.LeagueGameFinder(
        player_or_team_abbreviation="T", season_nullable=CURRENT, season_type_nullable="Regular Season",
        league_id_nullable="00", timeout=60).get_data_frames()[0], "team logs")
    tl["SEASON"] = CURRENT
    tl.to_csv(RAW / "team_game_logs_current.csv", index=False)
    print(f"Team games {CURRENT}: {len(tl) // 2}")

    rosters = []
    for t in teams.get_teams():
        r = with_retry(lambda: commonteamroster.CommonTeamRoster(
            team_id=t["id"], season=CURRENT, timeout=60).get_data_frames()[0], t["abbreviation"])
        rosters.append(r.assign(TEAM_ABBREVIATION=t["abbreviation"]))
        time.sleep(0.8)
    pd.concat(rosters, ignore_index=True).to_csv(RAW / "rosters_current.csv", index=False)
    print(f"Rosters: {sum(len(r) for r in rosters)} players")

    sch = with_retry(lambda: scheduleleaguev2.ScheduleLeagueV2(
        season=CURRENT, timeout=60).get_data_frames()[0], "schedule")
    sch.to_csv(RAW / "schedule_current.csv", index=False)

    subprocess.run([sys.executable, str(SRC / "fetch_injuries.py")], check=True)


if __name__ == "__main__":
    main()
