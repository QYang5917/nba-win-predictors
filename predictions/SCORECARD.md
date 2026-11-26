# Live prediction scorecard

Updated 2026-11-26. Model frozen on 2026-11-12. Game grading checks recorded prediction times precede tip-off; timestamps alone do not independently prove publication or input availability.

## Games: 91 graded of 1000 goal (0 pending)

Invalid game predictions excluded: 0.

- **Accuracy: 69.2%** (95% Wilson interval 59.1%–77.8%; assumes independent trials)
- Always picking the home team: 52.7%
- Log loss: 0.644 (lower is better; a coin flip is 0.693)

| Model's confidence | Games | Picks won |
|---|---|---|
| 50%–60% | 27 | 74.1% |
| 60%–70% | 19 | 73.7% |
| 70%–80% | 29 | 55.2% |
| 80%–100% | 16 | 81.2% |

## Player points: 1163 graded

- **Average miss: model 5.21 pts vs season average 5.26 pts** (same 1163 player-games)
- Scored over the prediction: 54.5% (about 50% means the median is right)
- Players who didn't play (void): 55

| Version | Player-games | Average miss | Scored over the prediction |
|---|---|---|---|
| v1 (frozen 2026-11-12) | 599 | 5.27 pts | 57.9% |
| v1.1 (frozen model, lineup fix 2026-11-19) | 564 | 5.15 pts | 50.9% |

Until November 19, 2026 the live run fed the points model too many players per team, which made its predictions about 1 point too low. Predictions labelled v1.1 use the same frozen model with that fixed.

The scores above measure accuracy, not profit. Pretend $1 bets against real sportsbook lines are logged in [paper_bets.csv](paper_bets.csv); their results will be reported here once about 1,000 have finished.
