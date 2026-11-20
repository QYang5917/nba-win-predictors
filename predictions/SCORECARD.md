# Live prediction scorecard

Updated 2026-11-20. Model frozen on 2026-11-12. Game grading checks recorded prediction times precede tip-off; timestamps alone do not independently prove publication or input availability.

## Games: 47 graded of 1000 goal (9 pending)

Invalid game predictions excluded: 0.

- **Accuracy: 66.0%** (95% Wilson interval 51.7%–77.8%; assumes independent trials)
- Always picking the home team: 44.7%
- Log loss: 0.661 (lower is better; a coin flip is 0.693)

| Model's confidence | Games | Picks won |
|---|---|---|
| 50%–60% | 14 | 71.4% |
| 60%–70% | 8 | 62.5% |
| 70%–80% | 18 | 61.1% |
| 80%–100% | 7 | 71.4% |

## Player points: 599 graded

- **Average miss: model 5.27 pts vs season average 5.32 pts** (same 599 player-games)
- Scored over the prediction: 57.9% (about 50% means the median is right)
- Players who didn't play (void): 31

Until November 19, 2026 the live run fed the points model too many players per team, which made its predictions about 1 point too low. Predictions labelled v1.1 use the same frozen model with that fixed.

The scores above measure accuracy, not profit. Pretend $1 bets against real sportsbook lines are logged in [paper_bets.csv](paper_bets.csv); their results will be reported here once about 1,000 have finished.
