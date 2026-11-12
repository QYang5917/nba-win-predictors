"""Team shot locations by season (rim, paint, mid-range, corner 3, above-the-break 3)."""
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashteamshotlocations

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2016, 2026)]

frames = []
for s in SEASONS:
    for attempt in range(4):
        try:
            df = leaguedashteamshotlocations.LeagueDashTeamShotLocations(
                season=s, distance_range="By Zone", timeout=60).get_data_frames()[0]
            break
        except Exception as exc:
            print(f"{s} retry ({exc})"); time.sleep(5 * (attempt + 1))
    df.columns = [" ".join(c).strip() if isinstance(c, tuple) else c for c in df.columns]
    frames.append(df.assign(SEASON=s))
    print(s, len(df)); time.sleep(1.2)
pd.concat(frames).to_csv(RAW / "team_shot_zones.csv", index=False)
