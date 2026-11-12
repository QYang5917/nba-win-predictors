#!/bin/zsh
# Live test: refresh data, grade finished games, predict games tipping soon, publish the log.
# Scheduled several times a day by scripts/com.nba-win-predictors.daily.plist.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
exec >> logs/daily.log 2>&1
echo "=== $(date) ==="
cd src
../.venv/bin/python fetch_daily.py
../.venv/bin/python grade_predictions.py
../.venv/bin/python predict_today.py
cd ..
git add predictions
if ! git diff --cached --quiet; then
  git commit -q -m "Live predictions $(date +%Y-%m-%d\ %H:%M)"
  git push -q
fi
