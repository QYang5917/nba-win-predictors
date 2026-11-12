"""Read the NBA's official injury reports (data/raw/injury_reports/*.pdf).

Each daily report lists only the teams playing that day, so we combine several days:
for each team we use the most recent report in which that team submitted a list.
Players on that list get its status; teammates not on it are treated as healthy.

The report gives today's status, not a return date, so injuries are grouped by the
reason text into rough expected absences:
    season-ending (fresh ACL / Achilles tear)       -> rest of season
    rehab (ACL / Achilles repair, from an older tear) -> ~40 games
    long-term (fracture, surgery, tear)              -> ~25 games
    short-term (strain, sprain, soreness)  -> ~3 games, "Questionable" -> ~1 game
G League / two-way assignments are ignored (those players rarely play anyway).
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "raw" / "injury_reports"
COLS = ["GameDate", "GameTime", "Matchup", "Team", "PlayerName", "CurrentStatus", "Reason"]
STATUSES = {"Out", "Doubtful", "Questionable", "Probable", "Available"}
SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}
PARTICLES = {"da", "de", "di", "van", "von", "la", "le", "del", "dos"}

SEASON_ENDING = re.compile(r"ACL|Achilles", re.I)
LONG_TERM = re.compile(r"Fracture|Surgery|Tear|Rupture|Repair|Meniscus|Reconstruct|Stress", re.I)
GAMES_OUT = {"season": 999, "rehab": 40, "long": 25, "short": 3, "questionable": 1}


def key(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def roster_key(full_name: str) -> str:
    """'Robert Williams III' -> key of 'WilliamsIII,Robert' (report order: Last, First)."""
    parts = full_name.split()
    if len(parts) > 2 and parts[-1].lower() in SUFFIXES:
        last, first = parts[-2] + parts[-1], " ".join(parts[:-2])
    elif len(parts) > 2 and parts[-2].lower() in PARTICLES:  # 'Tristan da Silva'
        last, first = parts[-2] + parts[-1], " ".join(parts[:-2])
    else:
        last, first = parts[-1], " ".join(parts[:-1])
    return key(last + first)


def parse_pdf(path: Path) -> pd.DataFrame:
    rows = []
    edges = None
    with pdfplumber.open(path) as pdf:
        team = None
        for page in pdf.pages:
            words = page.extract_words(keep_blank_chars=False, use_text_flow=False)
            header = {w["text"]: w for w in words if w["text"] in COLS}
            if len(header) == len(COLS):  # only page 1 has a header; reuse its columns
                edges = [header[c]["x0"] for c in COLS]
                skip_above = max(w["top"] for w in header.values()) + 2
            else:
                title = [w["top"] for w in words if w["text"] == "Report:"]
                skip_above = (min(title) + 2) if title else 0
            if edges is None:
                continue

            def col(x):
                i = sum(x >= e - 2 for e in edges) - 1
                return COLS[max(i, 0)]

            lines = {}
            for w in words:
                if w["top"] <= skip_above or w["text"].startswith("Page"):
                    continue
                lines.setdefault(round(w["top"]), []).append(w)
            entries, reasons = [], []
            for y, ws in sorted(lines.items()):
                cells = {}
                for w in sorted(ws, key=lambda w: w["x0"]):
                    cells.setdefault(col(w["x0"]), []).append(w["text"])
                if "Team" in cells:
                    team = "".join(cells["Team"])
                line_text = "".join(w["text"] for w in ws)
                name = "".join(cells.get("PlayerName", []))
                if "NOTYETSUBMITTED" in line_text:
                    entries.append({"y": y, "TEAM": team, "NAME": None, "STATUS": "NOT_SUBMITTED"})
                    continue
                if name and "CurrentStatus" in cells:
                    entries.append({"y": y, "TEAM": team, "NAME": name,
                                    "STATUS": "".join(cells["CurrentStatus"])})
                if "Reason" in cells:
                    reasons.append((y, "".join(cells["Reason"])))
            named = [e for e in entries if e["NAME"]]
            for e in entries:
                e["REASON"] = ""
            for y, text in reasons:  # wrapped reason fragments -> nearest player line
                if named:
                    min(named, key=lambda e: abs(e["y"] - y))["REASON"] += text
            rows += entries
    df = pd.DataFrame(rows).drop(columns="y")
    df["REPORT_DATE"] = path.stem
    return df


def categorize(status: str, reason: str) -> str | None:
    if "GLeague" in reason or status in ("Available", "Probable"):
        return None
    if status == "Out" and SEASON_ENDING.search(reason):
        return "rehab" if re.search(r"Repair|Surgery", reason, re.I) and \
            not re.search(r"Tear|Rupture", reason, re.I) else "season"
    if status == "Out" and LONG_TERM.search(reason):
        return "long"
    if status in ("Out", "Doubtful"):
        return "short"
    if status == "Questionable":
        return "questionable"
    return None


def current_injuries(rosters: pd.DataFrame) -> pd.DataFrame:
    """Latest injury status per rostered player: PLAYER_ID, STATUS, REASON, CATEGORY, GAMES_OUT."""
    reports = pd.concat([parse_pdf(p) for p in sorted(REPORTS.glob("*.pdf"))], ignore_index=True)
    team_names = {key(n): n for n in reports["TEAM"].dropna().unique()}
    submitted = reports[reports["STATUS"] != "NOT_SUBMITTED"]
    latest = submitted.groupby("TEAM")["REPORT_DATE"].max()
    cur = submitted[submitted["REPORT_DATE"] == submitted["TEAM"].map(latest)]
    cur = cur.dropna(subset=["NAME"]).copy()

    rk = rosters.assign(KEY=rosters["PLAYER"].map(roster_key))
    cur["KEY"] = cur["NAME"].map(key)
    out = cur.merge(rk[["KEY", "PLAYER_ID", "PLAYER", "TEAM_ABBREVIATION"]], on="KEY", how="left")
    unmatched = out[out["PLAYER_ID"].isna()]["NAME"].tolist()
    out = out.dropna(subset=["PLAYER_ID"])
    out["PLAYER_ID"] = out["PLAYER_ID"].astype(int)
    out["CATEGORY"] = [categorize(s, r) for s, r in zip(out["STATUS"], out["REASON"])]
    out = out.dropna(subset=["CATEGORY"])
    out["GAMES_OUT"] = out["CATEGORY"].map(GAMES_OUT)
    out.attrs["unmatched"] = unmatched
    out.attrs["teams_covered"] = len(latest)
    return out[["PLAYER_ID", "PLAYER", "TEAM_ABBREVIATION", "STATUS", "REASON",
                "CATEGORY", "GAMES_OUT", "REPORT_DATE"]].drop_duplicates("PLAYER_ID")


if __name__ == "__main__":
    rosters = pd.read_csv(ROOT / "data" / "raw" / "rosters_current.csv")
    inj = current_injuries(rosters)
    print(f"Teams covered: {inj.attrs['teams_covered']} / 30")
    print(f"Unmatched names (not on a current roster): {inj.attrs['unmatched']}")
    print(inj["CATEGORY"].value_counts().to_string())
    print(inj[inj["CATEGORY"].isin(["season", "long"])]
          .sort_values("CATEGORY")[["PLAYER", "TEAM_ABBREVIATION", "CATEGORY", "REASON"]].to_string(index=False))
