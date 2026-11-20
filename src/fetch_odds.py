"""Record sportsbook lines for today's games BEFORE tip-off, so the live test can measure profit.

Uses The Odds API (free plan: 500 credits a month). Lines are never used by the models; they are
only saved next to the predictions so each pick can later be compared with the price on offer.

  predictions/lines_games.csv   moneyline, spread and total for each game, one row per book and side
  predictions/lines_props.csv   player points over/under lines, one row per player and book

Each game's lines are saved once (the first successful pull) and never overwritten.
Credit budget: 3 per day for game lines + 1 per game for player points, about 300 a month.

The key is read from the ODDS_API_KEY environment variable or a `.env` file in the repo root
(`.env` is git-ignored, so the key is never committed).
"""
import datetime as dt
import json
import os
import re
import unicodedata
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from nba_api.stats.static import teams

ROOT = Path(__file__).resolve().parents[1]
RAW, PRED = ROOT / "data" / "raw", ROOT / "predictions"
STATE = ROOT / "logs" / "odds_state.json"  # how many times each pull was tried today
API = "https://api.the-odds-api.com/v4/sports/basketball_nba"
MIN_CREDITS_LEFT = 20       # stop before the monthly allowance runs out
MAX_GAME_PULLS_PER_DAY = 2  # each costs 3 credits
MAX_PROP_TRIES_PER_GAME = 8  # props may not be posted yet in the morning; an empty answer costs 0 credits
NICKNAME_TO_ABBR = {t["nickname"].lower(): t["abbreviation"] for t in teams.get_teams()}


def api_key() -> str | None:
    if os.environ.get("ODDS_API_KEY"):
        return os.environ["ODDS_API_KEY"].strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ODDS_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"") or None
    return None


def call(path: str, key: str, **params):
    """GET one endpoint. Returns (data, credits left). Never prints the URL, which holds the key."""
    try:
        r = requests.get(f"{API}{path}", params={"apiKey": key, **params}, timeout=30)
    except requests.RequestException as exc:
        raise RuntimeError(f"odds request failed ({type(exc).__name__})") from None
    if r.status_code != 200:
        raise RuntimeError(f"odds request returned status {r.status_code}")
    left = r.headers.get("x-requests-remaining")
    print(f"  odds call {path or '/'}: cost {r.headers.get('x-requests-last', '?')}, {left} credits left")
    return r.json(), (int(float(left)) if left else None)


def team_abbr(name: str) -> str | None:
    """'Los Angeles Clippers' or 'LA Clippers' -> 'LAC' (matched on the nickname)."""
    low = name.lower()
    return next((a for nick, a in NICKNAME_TO_ABBR.items() if low.endswith(" " + nick)), None)  # the space keeps Hornets from matching Nets


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s)
    return re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s.strip())


def todays_open_games(now: dt.datetime) -> pd.DataFrame:
    """Today's games (US Eastern date) that haven't tipped yet, from the NBA schedule."""
    sch = pd.read_csv(RAW / "schedule_current.csv", dtype={"gameId": str})
    sch = sch[sch["gameId"].str.startswith("002") & (sch["homeTeam_teamId"] > 0)].copy()
    sch["TIP_UTC"] = pd.to_datetime(sch["gameDateTimeUTC"], utc=True)
    today_et = now.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    g = sch[(sch["gameDateTimeEst"].str[:10] == today_et) & (sch["TIP_UTC"] > now)]
    return g.rename(columns={"gameId": "GAME_ID", "homeTeam_teamTricode": "HOME", "awayTeam_teamTricode": "AWAY"})[
        ["GAME_ID", "TIP_UTC", "HOME", "AWAY"]].assign(GAME_DATE=today_et).reset_index(drop=True)


def logged_ids(path: Path) -> set:
    return set(pd.read_csv(path, dtype={"GAME_ID": str})["GAME_ID"]) if path.exists() else set()


def append(path: Path, rows: list[dict]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def game_rows(event: dict, g, pulled: str) -> list[dict]:
    rows = []
    for book in event.get("bookmakers", []):
        for m in book.get("markets", []):
            for o in m.get("outcomes", []):
                if m["key"] == "totals":
                    side = o["name"].upper()  # OVER / UNDER
                else:
                    side = "HOME" if team_abbr(o["name"]) == g.HOME else "AWAY"
                rows.append(dict(PULLED_AT=pulled, GAME_DATE=g.GAME_DATE, GAME_ID=g.GAME_ID,
                                 TIP_UTC=g.TIP_UTC.strftime("%Y-%m-%dT%H:%M:%SZ"), HOME=g.HOME, AWAY=g.AWAY,
                                 BOOK=book["key"], MARKET=m["key"], SIDE=side,
                                 POINT=o.get("point"), PRICE=o["price"]))
    return rows


def prop_rows(event: dict, g, pulled: str, player_ids: dict) -> list[dict]:
    rows = []
    for book in event.get("bookmakers", []):
        for m in book.get("markets", []):
            if m["key"] != "player_points":
                continue
            lines = {}  # (player, line) -> {"Over": price, "Under": price}
            for o in m.get("outcomes", []):
                lines.setdefault((o.get("description"), o.get("point")), {})[o["name"]] = o["price"]
            for (player, point), px in lines.items():
                rows.append(dict(PULLED_AT=pulled, GAME_DATE=g.GAME_DATE, GAME_ID=g.GAME_ID,
                                 PLAYER_ID=player_ids.get(norm_name(player)), PLAYER_NAME=player,
                                 BOOK=book["key"], LINE=point,
                                 OVER_PRICE=px.get("Over"), UNDER_PRICE=px.get("Under")))
    return rows


def main():
    key = api_key()
    if not key:
        print("No odds key found (ODDS_API_KEY or .env); skipping betting lines.")
        return
    now = dt.datetime.now(dt.timezone.utc)
    pulled = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    games = todays_open_games(now)
    need_games = games[~games["GAME_ID"].isin(logged_ids(PRED / "lines_games.csv"))]
    need_props = games[~games["GAME_ID"].isin(logged_ids(PRED / "lines_props.csv"))]
    if need_games.empty and need_props.empty:
        print("Betting lines: nothing new to record.")
        return

    today = games["GAME_DATE"].iloc[0]
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if state.get("date") != today:
        state = {"date": today, "game_pulls": 0, "prop_tries": {}}
    left, events = None, None

    def match(event):
        home, away = team_abbr(event["home_team"]), team_abbr(event["away_team"])
        hit = games[(games["HOME"] == home) & (games["AWAY"] == away)]
        tip = pd.Timestamp(event["commence_time"])
        return hit.iloc[0] if len(hit) and tip > now else None  # only lines from before tip-off

    try:
        window = dict(commenceTimeFrom=pulled, commenceTimeTo=(games["TIP_UTC"].max() + pd.Timedelta(hours=1))
                      .strftime("%Y-%m-%dT%H:%M:%SZ"))
        if not need_games.empty and state["game_pulls"] < MAX_GAME_PULLS_PER_DAY:
            events, left = call("/odds", key, regions="us", markets="h2h,spreads,totals",
                                oddsFormat="decimal", **window)
            state["game_pulls"] += 1  # counted only when the call got through; a failed one costs no credits
            rows = []
            for ev in events:
                g = match(ev)
                if g is not None and g.GAME_ID in set(need_games["GAME_ID"]) and ev.get("bookmakers"):
                    rows += game_rows(ev, g, pulled)
            append(PRED / "lines_games.csv", rows)
            print(f"Betting lines: saved {len({r['GAME_ID'] for r in rows})} games ({len(rows)} rows).")

        if not need_props.empty and (left is None or left >= MIN_CREDITS_LEFT):
            if events is None:
                events, _ = call("/events", key, **window)  # free: just the list of games
            ros = pd.read_csv(RAW / "rosters_current.csv")
            player_ids = {norm_name(n): i for n, i in zip(ros["PLAYER"], ros["PLAYER_ID"])}
            for ev in events:
                g = match(ev)
                if g is None or g.GAME_ID not in set(need_props["GAME_ID"]):
                    continue
                if state["prop_tries"].get(g.GAME_ID, 0) >= MAX_PROP_TRIES_PER_GAME:
                    continue
                if left is not None and left < MIN_CREDITS_LEFT:
                    print("Betting lines: monthly credits nearly used up; stopping.")
                    break
                state["prop_tries"][g.GAME_ID] = state["prop_tries"].get(g.GAME_ID, 0) + 1
                data, left = call(f"/events/{ev['id']}/odds", key, regions="us", markets="player_points",
                                  oddsFormat="decimal")
                rows = prop_rows(data, g, pulled, player_ids)
                append(PRED / "lines_props.csv", rows)
                print(f"  {g.AWAY} @ {g.HOME}: {len(rows)} player points lines")
    finally:
        STATE.parent.mkdir(exist_ok=True)
        STATE.write_text(json.dumps(state))


if __name__ == "__main__":
    main()
