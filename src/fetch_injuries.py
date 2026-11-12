"""Download the latest official NBA injury report PDF for each of the last few days.

Reports are published at ak-static.cms.nba.com every 15 minutes on game days.
Each one lists only teams with upcoming games, so several days are combined later.
"""
import datetime as dt
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "injury_reports"
URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{}.pdf"
HEADERS = {"User-Agent": "Mozilla/5.0"}
DAYS_BACK = 5


def latest_for_day(day: dt.date) -> str | None:
    t = dt.datetime.combine(day, dt.time(23, 45))
    for _ in range(96):  # walk back in 15-minute steps
        url = URL.format(t.strftime("%Y-%m-%d_%I_%M%p"))
        if requests.head(url, headers=HEADERS, timeout=15).status_code == 200:
            return url
        t -= dt.timedelta(minutes=15)
    return None


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()
    for i in range(DAYS_BACK, -2, -1):
        day = today - dt.timedelta(days=i)
        url = latest_for_day(day)
        if url:
            (OUT / f"{day}.pdf").write_bytes(requests.get(url, headers=HEADERS, timeout=30).content)
            print(day, url.rsplit("_", 3)[-3:])
