# Live prediction scorecard

Updated 2026-11-21. Model frozen on 2026-11-12. Game grading checks recorded prediction times precede tip-off; timestamps alone do not independently prove publication or input availability.

## Games: 56 graded of 1000 goal (6 pending)

Invalid game predictions excluded: 0.

- **Accuracy: 66.1%** (95% Wilson interval 53.0%–77.1%; assumes independent trials)
- Always picking the home team: 46.4%
- Log loss: 0.648 (lower is better; a coin flip is 0.693)

| Model's confidence | Games | Picks won |
|---|---|---|
| 50%–60% | 15 | 66.7% |
| 60%–70% | 10 | 70.0% |
| 70%–80% | 22 | 59.1% |
| 80%–100% | 9 | 77.8% |

## Player points: 718 graded

- **Average miss: model 5.32 pts vs season average 5.36 pts** (same 718 player-games)
- Scored over the prediction: 56.5% (about 50% means the median is right)
- Players who didn't play (void): 33

| Version | Player-games | Average miss | Scored over the prediction |
|---|---|---|---|
| v1 (frozen 2026-11-12) | 599 | 5.27 pts | 57.9% |
| v1.1 (frozen model, lineup fix 2026-11-19) | 119 | 5.55 pts | 49.6% |

Until November 19, 2026 the live run fed the points model too many players per team, which made its predictions about 1 point too low. Predictions labelled v1.1 use the same frozen model with that fixed.

The scores above measure accuracy, not profit. Pretend $1 bets against real sportsbook lines are logged in [paper_bets.csv](paper_bets.csv); their results will be reported here once about 1,000 have finished.
