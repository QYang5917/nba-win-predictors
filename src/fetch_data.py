"""Step 1: Pull regular-season team game logs from stats.nba.com and save to CSV.

One row = one team in one game, so every game appears twice (once per team).
"""
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

# Last 10 complete regular seasons (2026-27 has only just started, so it's excluded).
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2016, 2026)]
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def fetch_season(season: str, retries: int = 3) -> pd.DataFrame:
    for attempt in range(1, retries + 1):
        try:
            finder = leaguegamefinder.LeagueGameFinder(
                player_or_team_abbreviation="T",
                season_nullable=season,
                season_type_nullable="Regular Season",
                league_id_nullable="00",  # NBA only (excludes WNBA/G League)
                timeout=60,
            )
            df = finder.get_data_frames()[0]
            df["SEASON"] = season
            return df
        except Exception as exc:  # network hiccups / rate limiting are common
            print(f"  {season}: attempt {attempt} failed ({exc}); retrying...")
            time.sleep(5 * attempt)
    raise RuntimeError(f"Could not fetch {season}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in SEASONS:
        df = fetch_season(season)
        print(f"{season}: {len(df):>5} team-game rows")
        frames.append(df)
        time.sleep(1.5)  # be polite to the API
    all_games = pd.concat(frames, ignore_index=True)
    out = OUT_DIR / "team_game_logs.csv"
    all_games.to_csv(out, index=False)
    print(f"Saved {len(all_games)} rows to {out}")


if __name__ == "__main__":
    main()
