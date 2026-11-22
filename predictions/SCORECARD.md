# Live prediction scorecard

Updated 2026-11-22. Model frozen on 2026-11-12. Game grading checks recorded prediction times precede tip-off; timestamps alone do not independently prove publication or input availability.

## Games: 62 graded of 1000 goal (7 pending)

Invalid game predictions excluded: 0.

- **Accuracy: 67.7%** (95% Wilson interval 55.4%–78.0%; assumes independent trials)
- Always picking the home team: 51.6%
- Log loss: 0.649 (lower is better; a coin flip is 0.693)

| Model's confidence | Games | Picks won |
|---|---|---|
| 50%–60% | 16 | 68.8% |
| 60%–70% | 12 | 75.0% |
| 70%–80% | 23 | 60.9% |
| 80%–100% | 11 | 72.7% |

## Player points: 791 graded

- **Average miss: model 5.35 pts vs season average 5.39 pts** (same 791 player-games)
- Scored over the prediction: 55.9% (about 50% means the median is right)
- Players who didn't play (void): 39

| Version | Player-games | Average miss | Scored over the prediction |
|---|---|---|---|
| v1 (frozen 2026-11-12) | 599 | 5.27 pts | 57.9% |
| v1.1 (frozen model, lineup fix 2026-11-19) | 192 | 5.59 pts | 49.5% |

Until November 19, 2026 the live run fed the points model too many players per team, which made its predictions about 1 point too low. Predictions labelled v1.1 use the same frozen model with that fixed.

The scores above measure accuracy, not profit. Pretend $1 bets against real sportsbook lines are logged in [paper_bets.csv](paper_bets.csv); their results will be reported here once about 1,000 have finished.
