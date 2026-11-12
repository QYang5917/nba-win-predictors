"""Player ages by season (used for the aging curve)."""
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2015, 2027)]

frames = []
for s in SEASONS:
    for attempt in range(4):
        try:
            df = leaguedashplayerstats.LeagueDashPlayerStats(season=s, timeout=60).get_data_frames()[0]
            break
        except Exception as exc:
            print(f"{s} retry ({exc})"); time.sleep(5 * (attempt + 1))
    frames.append(df[["PLAYER_ID", "PLAYER_NAME", "AGE"]].assign(SEASON=s))
    print(s, len(df)); time.sleep(1.2)
pd.concat(frames).to_csv(RAW / "player_ages.csv", index=False)
