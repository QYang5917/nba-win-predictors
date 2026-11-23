# Live prediction scorecard

Updated 2026-11-23. Model frozen on 2026-11-12. Game grading checks recorded prediction times precede tip-off; timestamps alone do not independently prove publication or input availability.

## Games: 69 graded of 1000 goal (0 pending)

Invalid game predictions excluded: 0.

- **Accuracy: 66.7%** (95% Wilson interval 54.9%–76.6%; assumes independent trials)
- Always picking the home team: 53.6%
- Log loss: 0.659 (lower is better; a coin flip is 0.693)

| Model's confidence | Games | Picks won |
|---|---|---|
| 50%–60% | 19 | 68.4% |
| 60%–70% | 15 | 73.3% |
| 70%–80% | 24 | 58.3% |
| 80%–100% | 11 | 72.7% |

## Player points: 872 graded

- **Average miss: model 5.38 pts vs season average 5.42 pts** (same 872 player-games)
- Scored over the prediction: 55.2% (about 50% means the median is right)
- Players who didn't play (void): 45

| Version | Player-games | Average miss | Scored over the prediction |
|---|---|---|---|
| v1 (frozen 2026-11-12) | 599 | 5.27 pts | 57.9% |
| v1.1 (frozen model, lineup fix 2026-11-19) | 273 | 5.63 pts | 49.1% |

Until November 19, 2026 the live run fed the points model too many players per team, which made its predictions about 1 point too low. Predictions labelled v1.1 use the same frozen model with that fixed.

The scores above measure accuracy, not profit. Pretend $1 bets against real sportsbook lines are logged in [paper_bets.csv](paper_bets.csv); their results will be reported here once about 1,000 have finished.
