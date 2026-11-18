# Live prediction scorecard

Updated 2026-11-18. Model frozen on 2026-11-12. Game grading checks recorded prediction times precede tip-off; timestamps alone do not independently prove publication or input availability.

## Games: 31 graded of 1000 goal (0 pending)

Invalid game predictions excluded: 0.

- **Accuracy: 71.0%** (95% Wilson interval 53.4%–83.9%; assumes independent trials)
- Always picking the home team: 51.6%
- Log loss: 0.618 (lower is better; a coin flip is 0.693)

| Model's confidence | Games | Picks won |
|---|---|---|
| 50%–60% | 10 | 90.0% |
| 60%–70% | 5 | 40.0% |
| 70%–80% | 11 | 54.5% |
| 80%–100% | 5 | 100.0% |

## Player points: 397 graded

- **Average miss: model 5.37 pts vs season average 5.45 pts** (same 397 player-games)
- Scored over the prediction: 56.2% (about 50% means the median is right)
- Players who didn't play (void): 18

The scores above measure accuracy, not profit. Profit is tracked under paper bets.

## Paper bets (no real money)

A pretend $1 bet is logged before tip-off whenever the frozen model's chance times the best recorded price shows positive value. The main rule, fixed in advance, is value above 5%.

![Paper bets: running profit, and win rate against the break-even rate](paper_bets.png)

### Summary

| | All paper bets | Main rule (value above 5%) |
|---|---|---|
| Bets logged | 202 | 174 |
| Settled | 63 | 56 |
| Waiting for results | 137 | 117 |
| Void (player didn't play) | 2 | 1 |
| Won | 24 | 22 |
| Lost | 39 | 34 |
| Win rate | 38.1% | 39.3% |
| Win rate needed to break even | 38.4% | 37.8% |
| Average odds (payout per $1) | 2.70 | 2.74 |
| Money staked | $63 | $56 |
| Profit or loss | +$0.66 | +$2.71 |
| Return per $1 | +1.0% | +4.8% |

Could the main rule's result be luck? yes (±36%).

### By market and rule

| Market | Rule | Settled | Won | Win rate | Break-even | Profit or loss | Return per $1 | Could be luck? |
|---|---|---|---|---|---|---|---|---|
| moneyline | any positive value | 5 | 2 | 40.0% | 44.0% | +$1.19 | +23.8% | too few bets to say |
| moneyline | **value above 5% (main rule)** | 4 | 2 | 50.0% | 44.1% | +$2.19 | +54.7% | too few bets to say |
| moneyline | value above 10% | 2 | 0 | 0.0% | 48.9% | -$2.00 | -100.0% | too few bets to say |
| points | any positive value | 58 | 22 | 37.9% | 37.9% | -$0.53 | -0.9% | yes (±34%) |
| points | **value above 5% (main rule)** | 52 | 20 | 38.5% | 37.3% | +$0.52 | +1.0% | yes (±36%) |
| points | value above 10% | 42 | 15 | 35.7% | 36.5% | -$1.85 | -4.4% | yes (±41%) |

### Day by day (all paper bets, last 14 days)

| Date | Settled | Won | Win rate | Profit or loss | Running total |
|---|---|---|---|---|---|
| 2026-11-17 | 63 | 24 | 38.1% | +$0.66 | +$0.66 |

### Latest settled bets

| Date | Bet | Book | Odds | Model's chance | Result | Profit or loss |
|---|---|---|---|---|---|---|
| 2026-11-17 | Keaton Wagler under 9.5 points | bovada | 2.60 | 42% | lost | -$1.00 |
| 2026-11-17 | Darryn Peterson under 18.5 points | bovada | 2.90 | 62% | won | +$1.90 |
| 2026-11-17 | Ace Bailey over 13.5 points | betrivers | 1.83 | 55% | lost | -$1.00 |
| 2026-11-17 | Brice Sensabaugh under 10.5 points | bovada | 3.00 | 52% | lost | -$1.00 |
| 2026-11-17 | Keyonte George over 21.5 points | betmgm | 1.95 | 53% | won | +$0.95 |
| 2026-11-17 | Darius Garland under 19.5 points | bovada | 3.05 | 42% | lost | -$1.00 |
| 2026-11-17 | Rui Hachimura under 11.5 points | bovada | 2.75 | 49% | lost | -$1.00 |
| 2026-11-17 | Josh Okogie under 4.5 points | bovada | 2.50 | 49% | lost | -$1.00 |
| 2026-11-17 | Derrick Jones Jr. over 9.5 points | betonlineag | 1.81 | 58% | lost | -$1.00 |
| 2026-11-17 | Kris Dunn over 7.5 points | bovada | 2.90 | 43% | lost | -$1.00 |

Break-even is the win rate needed to make exactly $0 at the odds taken. Points lines can be recorded a few hours after the prediction, so some bets use an older prediction against a newer line. The full list is in [paper_bets.csv](paper_bets.csv).
