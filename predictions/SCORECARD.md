# Live prediction scorecard

Updated 2026-11-17. Model frozen on 2026-11-12. Game grading checks recorded prediction times precede tip-off; timestamps alone do not independently prove publication or input availability.

## Games: 29 graded of 1000 goal (2 pending)

Invalid game predictions excluded: 0.

- **Accuracy: 72.4%** (95% Wilson interval 54.3%–85.3%; assumes independent trials)
- Always picking the home team: 58.6%
- Log loss: 0.561 (lower is better; a coin flip is 0.693)

| Model's confidence | Games | Picks won |
|---|---|---|
| 50%–60% | 10 | 80.0% |
| 60%–70% | 4 | 25.0% |
| 70%–80% | 10 | 70.0% |
| 80%–100% | 5 | 100.0% |

## Player points: 332 graded

- **Average miss: model 5.32 pts vs season average 5.44 pts** (same 332 player-games)
- Scored over the prediction: 57.8% (about 50% means the median is right)
- Players who didn't play (void): 55

The scores above measure accuracy, not profit. Profit is tracked under paper bets.

## Paper bets (no real money)

A pretend $1 bet is logged before tip-off whenever the frozen model's chance times the best recorded price shows positive value. The main rule, fixed in advance, is value above 5%.

| Market | Rule | Bets | Won | Return per $1 | Could be luck? |
|---|---|---|---|---|---|
| moneyline | any positive value | 3 | 66.7% | +35.7% | too few bets to say |
| moneyline | **value above 5% (main rule)** | 2 | 50.0% | -10.5% | too few bets to say |
| moneyline | value above 10% | 0 | – | – | – |
| points | any positive value | 0 | – | – | – |
| points | **value above 5% (main rule)** | 0 | – | – | – |
| points | value above 10% | 0 | – | – | – |

Waiting for results: 62. Void (player didn't play): 0. Points lines can be recorded a few hours after the prediction, so some bets use an older prediction against a newer line.
