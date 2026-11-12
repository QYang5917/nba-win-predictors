"""Charts for the README: projected wins and championship odds for 2026-27."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, INK, INK2, GRID, BG = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.dpi": 160})

odds = pd.read_csv(ROOT / "data" / "processed" / "championship_odds_2026-27.csv").head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 5))
colors = [BLUE if c == "East" else ORANGE for c in odds["CONF"]]
ax.barh(odds["TEAM"], odds["CHAMPION"], color=colors, height=0.6)
for y, v in enumerate(odds["CHAMPION"]):
    ax.text(v + 0.003, y, f"{v:.1%}", va="center", color=INK2, fontsize=9)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel("Chance of winning the 2027 title (20,000 simulated seasons)", color=INK2)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
          labels=["East", "West"], frameon=False, loc="lower right")
fig.suptitle("Boston and San Antonio lead, but no team is above 25%", x=0.02, ha="left",
             color=INK, fontweight="bold")
fig.tight_layout(); fig.savefig(ROOT / "figures" / "championship_odds.png")

proj = pd.read_csv(ROOT / "data" / "processed" / "championship_odds_2026-27.csv").sort_values("PROJ_WINS")
fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharex=True)
for ax, conf, color in [(axes[0], "East", BLUE), (axes[1], "West", ORANGE)]:
    d = proj[proj["CONF"] == conf]
    ax.hlines(d["TEAM"], d["WINS_P10"], d["WINS_P90"], color=GRID, lw=6)
    ax.plot(d["PROJ_WINS"], d["TEAM"], "o", color=color, ms=8)
    ax.axvline(41, color=INK2, lw=0.8, ls="--")
    ax.set_title(conf, loc="left", color=INK, fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("Projected wins (dot) and 80% range (bar)", color=INK2)
fig.suptitle("2026-27 projected wins as of Nov 11", x=0.02, ha="left", color=INK, fontweight="bold")
fig.tight_layout(); fig.savefig(ROOT / "figures" / "projected_wins.png")
print("saved")
