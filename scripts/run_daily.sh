#!/bin/zsh
# Live test: refresh data, grade finished games, predict games tipping soon, publish the log.
# Scheduled several times a day by scripts/com.nba-win-predictors.daily.plist.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
exec >> logs/daily.log 2>&1
echo "=== $(date) ==="
cd src
../.venv/bin/python fetch_daily.py || echo "WARNING: data refresh failed; using cached data."
../.venv/bin/python grade_predictions.py
../.venv/bin/python predict_today.py
../.venv/bin/python fetch_odds.py || echo "WARNING: betting lines not recorded this run."
../.venv/bin/python paper_bets.py || echo "WARNING: paper bets not updated this run."
cd ..
git add predictions
if ! git diff --cached --quiet; then
  git commit -q -m "Live predictions $(date +%Y-%m-%d\ %H:%M)"
  git push -q
fi
