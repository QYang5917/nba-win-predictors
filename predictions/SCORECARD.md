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

![Paper bets: running profit, and win rate against the break-even rate](paper_bets.png)

### Summary

| | All paper bets | Main rule (value above 5%) |
|---|---|---|
| Bets logged | 65 | 57 |
| Settled | 3 | 2 |
| Waiting for results | 62 | 55 |
| Void (player didn't play) | 0 | 0 |
| Won | 2 | 1 |
| Lost | 1 | 1 |
| Win rate | 66.7% | 50.0% |
| Win rate needed to break even | 40.8% | 39.3% |
| Average odds (payout per $1) | 2.82 | 3.10 |
| Money staked | $3 | $2 |
| Profit or loss | +$1.07 | -$0.21 |
| Return per $1 | +35.7% | -10.5% |

Could the main rule's result be luck? too few bets to say.

### By market and rule

| Market | Rule | Settled | Won | Win rate | Break-even | Profit or loss | Return per $1 | Could be luck? |
|---|---|---|---|---|---|---|---|---|
| moneyline | any positive value | 3 | 2 | 66.7% | 40.8% | +$1.07 | +35.7% | too few bets to say |
| moneyline | **value above 5% (main rule)** | 2 | 1 | 50.0% | 39.3% | -$0.21 | -10.5% | too few bets to say |
| moneyline | value above 10% | 0 | – | – | – | – | – | – |
| points | any positive value | 0 | – | – | – | – | – | – |
| points | **value above 5% (main rule)** | 0 | – | – | – | – | – | – |
| points | value above 10% | 0 | – | – | – | – | – | – |

### Day by day (all paper bets, last 14 days)

| Date | Settled | Won | Win rate | Profit or loss | Running total |
|---|---|---|---|---|---|
| 2026-11-17 | 3 | 2 | 66.7% | +$1.07 | +$1.07 |

### Latest settled bets

| Date | Bet | Book | Odds | Model's chance | Result | Profit or loss |
|---|---|---|---|---|---|---|
| 2026-11-17 | DET to win | betus | 4.40 | 24% | lost | -$1.00 |
| 2026-11-17 | MIN to win | fanduel | 1.79 | 60% | won | +$0.79 |
| 2026-11-17 | TOR to win | fanduel | 2.28 | 44% | won | +$1.28 |

Break-even is the win rate needed to make exactly $0 at the odds taken. Points lines can be recorded a few hours after the prediction, so some bets use an older prediction against a newer line. The full list is in [paper_bets.csv](paper_bets.csv).
