# Live prediction scorecard

Updated 2026-11-19. Model frozen on 2026-11-12. Game grading checks recorded prediction times precede tip-off; timestamps alone do not independently prove publication or input availability.

## Bets so far (pretend $1 bets, no real money)

| | Bets placed | Finished | Won | Lost | Win rate | Profit or loss |
|---|---|---|---|---|---|---|
| Game winner | 20 | 16 | 6 | 10 | 37.5% | -$2.53 |
| Player points | 225 | 176 | 63 | 113 | 35.8% | -$10.28 |
| **Total** | **245** | **192** | **69** | **123** | **35.9%** | **-$12.81** |

Bets that are not finished are waiting for the game to end. Details are under [Paper bets](#paper-bets-no-real-money) below.

## Games: 43 graded of 1000 goal (4 pending)

Invalid game predictions excluded: 0.

- **Accuracy: 65.1%** (95% Wilson interval 50.2%–77.6%; assumes independent trials)
- Always picking the home team: 44.2%
- Log loss: 0.679 (lower is better; a coin flip is 0.693)

| Model's confidence | Games | Picks won |
|---|---|---|
| 50%–60% | 13 | 76.9% |
| 60%–70% | 7 | 57.1% |
| 70%–80% | 16 | 56.2% |
| 80%–100% | 7 | 71.4% |

## Player points: 553 graded

- **Average miss: model 5.28 pts vs season average 5.35 pts** (same 553 player-games)
- Scored over the prediction: 57.3% (about 50% means the median is right)
- Players who didn't play (void): 26

Until November 19, 2026 the live run fed the points model too many players per team, which made its predictions about 1 point too low. Predictions labelled v1.1 use the same frozen model with that fixed.

The scores above measure accuracy, not profit. Profit is tracked under paper bets.

## Paper bets (no real money)

A pretend $1 bet is logged before tip-off whenever the frozen model's chance times the best recorded price shows positive value. The main rule, fixed in advance, is value above 5%.

![Paper bets: running profit, and win rate against the break-even rate](paper_bets.png)

### Summary

| | All paper bets | Main rule (value above 5%) |
|---|---|---|
| Bets logged | 245 | 208 |
| Settled | 192 | 167 |
| Waiting for results | 43 | 34 |
| Void (player didn't play) | 10 | 7 |
| Won | 69 | 61 |
| Lost | 123 | 106 |
| Win rate | 35.9% | 36.5% |
| Win rate needed to break even | 38.8% | 37.6% |
| Average odds (payout per $1) | 2.69 | 2.75 |
| Money staked | $192 | $167 |
| Profit or loss | -$12.81 | -$4.56 |
| Return per $1 | -6.7% | -2.7% |

Could the main rule's result be luck? yes (±20%).

### By market and rule

| Market | Rule | Settled | Won | Win rate | Break-even | Profit or loss | Return per $1 | Could be luck? |
|---|---|---|---|---|---|---|---|---|
| moneyline | any positive value | 16 | 6 | 37.5% | 52.8% | -$2.53 | -15.8% | too few bets to say |
| moneyline | **value above 5% (main rule)** | 10 | 4 | 40.0% | 46.5% | +$0.73 | +7.3% | too few bets to say |
| moneyline | value above 10% | 8 | 2 | 25.0% | 48.4% | -$3.46 | -43.2% | too few bets to say |
| points | any positive value | 176 | 63 | 35.8% | 37.6% | -$10.28 | -5.8% | yes (±19%) |
| points | **value above 5% (main rule)** | 157 | 57 | 36.3% | 37.0% | -$5.29 | -3.4% | yes (±21%) |
| points | value above 10% | 136 | 47 | 34.6% | 36.3% | -$7.98 | -5.9% | yes (±23%) |

### Day by day (all paper bets, last 14 days)

| Date | Settled | Won | Win rate | Profit or loss | Running total |
|---|---|---|---|---|---|
| 2026-11-18 | 129 | 45 | 34.9% | -$13.47 | -$12.81 |
| 2026-11-17 | 63 | 24 | 38.1% | +$0.66 | +$0.66 |

### Latest settled bets

| Date | Bet | Book | Odds | Model's chance | Result | Profit or loss |
|---|---|---|---|---|---|---|
| 2026-11-18 | Darius Acuff Jr. under 20.5 points | bovada | 3.00 | 57% | lost | -$1.00 |
| 2026-11-18 | Oso Ighodaro under 4.5 points | bovada | 2.80 | 36% | lost | -$1.00 |
| 2026-11-18 | Collin Gillespie under 9.5 points | bovada | 2.65 | 47% | won | +$1.65 |
| 2026-11-18 | Jordan Goodwin under 6.5 points | bovada | 2.75 | 38% | won | +$1.75 |
| 2026-11-18 | Jalen Green under 10.5 points | bovada | 3.15 | 36% | lost | -$1.00 |
| 2026-11-18 | Precious Achiuwa under 6.5 points | bovada | 2.85 | 44% | won | +$1.85 |
| 2026-11-18 | De'Andre Hunter under 11.5 points | bovada | 2.65 | 43% | won | +$1.65 |
| 2026-11-18 | Miles Bridges under 10.5 points | bovada | 3.05 | 37% | lost | -$1.00 |
| 2026-11-18 | Dillon Brooks under 11.5 points | bovada | 2.75 | 39% | lost | -$1.00 |
| 2026-11-18 | Domantas Sabonis under 13.5 points | betonlineag | 1.97 | 54% | won | +$0.97 |

Break-even is the win rate needed to make exactly $0 at the odds taken. Points lines can be recorded a few hours after the prediction, so some bets use an older prediction against a newer line. The full list is in [paper_bets.csv](paper_bets.csv).
