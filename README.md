# NBA Win Predictors: what wins games, and who will win this season

**V2 follow-up:** [Past-only calibration, learned blending and recency selection](reports/V2_FOLLOWUP.md)
do not beat the previous fixed blend's 68.46% historical accuracy. A past-selected linear
regularization setting changes log loss only from 0.591566 to 0.591550, with no extra
correct picks and uncertainty spanning zero. Optional research inference bundles are saved.

**V2 development:** [Chronological nonlinear and margin-model experiment](reports/V2_RESEARCH.md).
A fixed linear/boosting blend scores 68.46% versus a 68.18% causal player baseline
on 3,633 common games (10 extra correct picks), with uncertainty including no gain.
Standalone boosting models perform worse. Research models and a separate inference
entry point are saved; possession-level player impact and live integration remain pending.

**Impact-system follow-up:** [Advanced-input normalization, rating shrinkage and context tests](reports/IMPACT_SYSTEM_UPDATE.md).
Stronger player-impact shrinkage worsens historical probability loss. Adding team context
reduces log loss from 0.594113 to 0.592389, while accuracy declines from 68.26% to 68.15%.
Neither is deployed. NBA advanced-data retrieval remains blocked by DNS, so a fully trained
advanced-impact ranking is still pending real inputs and historical targets.

**Player-rating experiment:** [Continuous recency weighting and joint home estimation](reports/RATING_EXPERIMENT.md)
compare four rating methods with identical pregame candidates/minutes. On 3,633 common
games, recency weighting changes accuracy from 68.07% to 68.18% (four extra correct picks)
and log loss from 0.595332 to 0.593935. The last season's accuracy falls, and the historical
evaluation period has been reused; this is a research candidate, not a deployed upgrade.
A follow-up preserves total training weight while emphasizing recent games: accuracy
remains 68.18%, with log loss 0.593753. Its paired difference from the first recency
variant includes zero, so superiority over that variant is not established.

**2026-11-13 data audit:** [Evidence, data defects and evaluation limitations](reports/DATA_AUTHENTICITY_AUDIT.md).
Official game spot checks support real-world scores, but do not authenticate every saved row.
Among 11,979 historical games, 129 have inconsistent team `PLUS_MINUS` fields and 10 have
ambiguous home/away markers. The 68.3% actual-participant result below has hindsight
information; four live wins do not establish 100% accuracy. Reproduce the read-only
consistency audit with `.venv/bin/python src/audit_data.py`.

**2026-11-12 research update:** A separate [walk-forward market evaluation](reports/MARKET_RESEARCH.md)
now compares the previous-game player model, market, calibrated market and a market/player hybrid.
It uses exact-date odds matching and trains combination weights only on earlier seasons.
On 3,256 evaluation games the hybrid has log loss 0.59006 versus the market's 0.59014,
but its 22 flat-unit simulated bets lose 3.6%. This is not evidence of a profitable edge.
The historical notebook below remains an exploratory analysis; frozen live models are unchanged.

A subsequent [team-context experiment](reports/CONTEXT_EXPERIMENT.md) adds causal Elo,
recent net efficiency and schedule load to the previous-game player model. Across the
same 3,256 games, log loss improves from 0.60807 to 0.60450 in all three evaluation
seasons, but simulated ROI remains negative (-5.12%). This is a research candidate,
not a deployed or profitable model. Reproduce with `.venv/bin/python src/context_experiment.py`.

The next [rotation experiment](reports/ROTATION_RESEARCH.md) reduces historical
minute-prediction MAE from 7.30 to 7.00 using a learned median model; the learned
mean model reduces RMSE from 10.73 to 10.26. Forced 240-minute normalization fails
and is not adopted. Complete-availability scenarios and timestamped snapshot
ingestion are available for research; these results do not establish better win
predictions or betting returns.

An [end-to-end rotation/win experiment](reports/ROTATION_WIN_EXPERIMENT.md) now
cross-fits minutes, rebuilds historical player ratings, and trains win/market models
only on earlier seasons. On 3,209 common games, learned mean minutes slightly lower
log loss (0.60612 → 0.60590) and simulated losses (-4.07% → -3.36%), but the paired
uncertainty interval includes no improvement. All branches, including a small-sample
profitable legacy/market branch, are reported; none establishes stable profitability.

The [five-game candidate-pool experiment](reports/AVAILABILITY_EXPERIMENT.md)
raises actual-participant coverage from 85.63% to 96.52%. Learned minutes reduce
RMSE on a common expanded evaluation set, but MAE worsens slightly. The direct
minute variant gives log loss 0.60583 and simulated ROI -3.14% on the same 3,209
games; neither the probability change nor positive small-bet branches establishes
a reliable edge. Participation modeling remains research-only.

I analyzed **11,979 NBA regular-season games across 10 seasons (2016-17 to 2025-26)**. Every model trains on the first seven seasons (8,289 games) and is tested on the three most recent (**2023-24, 2024-25, 2025-26**: 3,690 games). The split is by season, never random.

## Results at a glance

| Question | Model | Train accuracy | Test accuracy |
|---|---|---|---|
| — | Baseline: always pick the home team | 56.9% | **54.7%** |
| **Who will win?** (pregame info only) | Logistic regression (team averages) | 64.4% | **66.9%** |
| | Random forest (team averages) | 66.6% | **66.3%** |
| | Player-based strength (actual participants; hindsight information) | 65.7% | 68.3% |
| | Player-based strength (last game's players; historical estimate) | 65.1% | 67.9% |
| **What explains a win?** (stats from the game itself) | Logistic regression | 95.7% | 96.0% |
| | Random forest | 94.6% | 93.0% |

**Why two questions?** An audit found that same-game shooting, free throw, turnover and rebounding rates rebuild a team's points per possession with 98% accuracy, so they come close to revealing the final score. The 96% models explain completed games. The 68.3% player result also uses hindsight through actual participants, so it is not an independently validated pregame accuracy. Previous-game participants remove that particular leak, but historical results still require the data-quality and evaluation checks described in the audit.

## Key findings: what explains a win

1. **Shooting efficiency decides games.** The team with the better effective field goal % (a shooting percentage that counts a three as 1.5 makes) wins most of the time. Remove it from the model and accuracy drops by about 35 percentage points, more than every other stat combined.
2. **After shooting: turnovers, offensive rebounds, free throws, in that order.** This matches basketball analyst Dean Oliver's "Four Factors" theory, now confirmed on ten years of recent data.
3. **Pace, 3-point volume and assists barely matter on their own.** Taking *more* threes doesn't win games; *making* shots does. Playing fast or slow has no effect on who wins.
4. **Home court works mostly through the stats.** Home teams win 56% of games. Give both teams identical stats and the home team still wins about 52%, so most of the edge shows up *as* better shooting, with only a small extra bump on top.
5. **A simple model is enough.** Logistic regression (96% accuracy) beat a random forest (93%). The relationships are mostly straight lines, so the more complex model had nothing extra to find.

![Feature importance](figures/permutation_importance.png)

![Logistic regression coefficients](figures/logreg_coefficients.png)

![Model accuracy](figures/model_accuracy.png)

## Key findings: forecasting before tip-off
- **Net rating so far is the best single predictor** (points scored minus allowed per 100 possessions). It already combines all Four Factors.
- **Top 5 pregame features** (logistic regression, standardized coefficients): net rating +0.23, win % +0.23, shooting (eFG%) +0.19, rest days +0.11, defensive rebound % +0.10.
- **Surprise 1: the test seasons were *easier* to predict than the training seasons** (66.9% vs 64.4%). Recent seasons had bigger gaps between the best and worst teams (the spread of team win % rose from 0.14 to 0.165), so favorites won more often.
- **Surprise 2: rest days have a clear coefficient but add almost no accuracy.** A rested team wins slightly more often, but rest rarely changes *who is favored*, so it seldom flips a prediction.
- **Overfitting caught and fixed:** the first random forest scored 74.5% on training games but 66.1% on test games. Requiring each leaf to cover 100+ games closed the gap (66.6% vs 66.3%) with no loss on test.

## Predicting games from players, not team averages
Team averages can't tell when a star sits out. So each team's pregame strength is built from its players: each player's rating (Part 2) × expected minutes, using only information from before the game. Ratings are refit every 14 days from earlier games only.

![Pregame accuracy](figures/pregame_accuracy.png)

- **Players beat team averages:** 68.3% vs 66.9%, and log loss (a check that the predicted *probabilities* are right) improves from 0.607 to 0.588.
- **Honest about leakage:** "tonight's lineup" uses who actually played, which in real use comes from the injury report and starting lineups shortly before tip-off. Coach's-decision benchings aren't known in advance, so I also built a zero-leakage version from the previous game's players. It still reaches 67.9%.
- **Near the ceiling:** with hindsight, knowing every team's full-season quality only reaches 68.9%. The rest is game-to-game luck no pregame model can see.

## Live test: 1,000 games predicted before tip-off
Backtests can hide mistakes, so the models are now **frozen** (trained on 2016-17 to 2025-26) and predicting 2026-27 games live, starting November 12, 2026. On each game day, the first scheduled run that gets through (usually in the morning) reads the latest official injury report, logs a win probability for the game and a points prediction for every player expected to play 20+ minutes, then commits the log to GitHub. The commit timestamps prove each prediction was made before tip-off.

Since November 17, 2026 the same run also saves the sportsbook lines on offer (moneyline, spread, total and player points, from several books via The Odds API) and logs **paper bets**: a pretend $1 bet whenever the model's chance times the best recorded price shows positive value. No real money is involved. The rule that gets judged (value above 5%) was fixed before the first bet, and the [scorecard](predictions/SCORECARD.md) reports the return per $1 next to the accuracy.

- **Scorecard:** [`predictions/SCORECARD.md`](predictions/SCORECARD.md) (accuracy, calibration, points error), updated automatically
- **Raw logs:** [`predictions/games.csv`](predictions/games.csv) and [`predictions/props.csv`](predictions/props.csv)
- **Rules:** the models don't change during the test. Any idea for improving them gets written down first and tested at the end.
- **Code:** [`src/predict_today.py`](src/predict_today.py), [`src/grade_predictions.py`](src/grade_predictions.py), [`scripts/run_daily.sh`](scripts/run_daily.sh)

## Can it beat the betting market? No, and here's the proof
Picking winners isn't how betting pays: favorites pay little, and point-spread bets need 52.4% wins to break even at standard -110 odds. So I tested the model against **real BetMGM closing odds** on 3,257 test games (2023-24 through Feb 2026):

| Who | Picked the winner | Log loss (lower = better) |
|---|---|---|
| **Betting market** | **68.4%** | **0.590** |
| Our model (tonight's lineup) | 67.2% | 0.601 |
| Our model (zero leakage) | 67.0% | 0.608 |

- **The market wins.** Closing odds already include everything the model knows, plus late injury news and money from professional bettors.
- **The model adds almost nothing on top.** Combining the two puts 0.95 of the weight on the market and 0.09 on the model.
- **Betting on it loses money.** Betting only when the model disagrees with the odds lost 0.6% to 10.5% in 15 of 16 settings. The one profitable setting (+0.6% on 281 bets) is the best of 16 tries and well within luck.
- **Why it matters:** 68% accuracy sounds impressive, but the market is the real bar. Testing against it separates a real edge from a good-looking number.

## Testing a fan's hypothesis: is the modern NBA all about the 3?
From watching games, my theory was: score in fewer seconds than the opponent, and favor 3s over 2s over mid-range shots. The data says I was **mostly right**, with one surprise:

| The team that… | Won this often |
|---|---|
| Played faster (more possessions) | ~42% |
| Took more 3s | ~50% (coin flip) |
| Made more 3s | ~66% |
| Had the higher 3P% | ~71% |
| Scored more points per possession | **~96%** |

- **Efficiency, not speed.** Playing fast doesn't win. Scoring more *per trip* does.
- **Making 3s matters; taking them doesn't.**
- **Shot value: rim > corner 3 > above-break 3 > paint > mid-range.** Mid-range really is the worst shot, and teams have cut it in half since 2016-17. The surprise: shots at the rim are worth even more than 3s.
- **Corner 3s are the shot mix most linked to winning** (correlation 0.25 with win %, within each season).
- **No forecasting gain.** Adding 3-point stats to the pregame model moved accuracy from 66.9% to 67.0%, because net rating already captures shooting.

![Points per shot by zone](figures/points_per_shot_by_zone.png)

## How it works

| Step | What happens | Where |
|---|---|---|
| 1. Data | Download every team box score for 10 seasons from stats.nba.com with `nba_api` | [`src/fetch_data.py`](src/fetch_data.py) |
| 2. Features | Turn raw counts into rates (per possession or per shot), then take **home minus away** for each game | [`src/build_features.py`](src/build_features.py) |
| 3. Models | Logistic regression and random forest. **Train on 2016-17 to 2022-23, test on 2023-24 to 2025-26**, with no random shuffling | [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) |
| 4. Importance | Standardized coefficients plus permutation importance on the test set | same notebook |
| 5. Pregame model | Each team's season-to-date averages *before* each game, blended with last season early on | [`src/build_pregame_features.py`](src/build_pregame_features.py) |
| 6. Player-based model | Team strength from player ratings × expected minutes, before each game | [`src/build_player_strength.py`](src/build_player_strength.py), notebook Part 1d |
| 7. Market test | Compare win probabilities with BetMGM closing odds and simulate bets | notebook Part 1e |
| 8. Shot value | Points per shot by court zone and how shot mix relates to winning | [`src/fetch_shot_zones.py`](src/fetch_shot_zones.py), notebook Part 1c |

### Design choices worth noting
- **No direct score leakage.** Points, points allowed, point differential and plus-minus are never used as same-game features. The indirect leak (shooting + free throws + turnovers + rebounds ≈ points) is why the forecasting question uses pregame data only.
- **Rates, not totals.** Fast teams take more shots *and* commit more turnovers just because they have more possessions. Dividing by possessions makes teams comparable.
- **One row per game.** The raw data lists each game twice (once per team). Keeping both copies would double-count every game.
- **Time-based split.** Testing on later seasons copies real use and checks that the findings still hold as the league changes (for example, more threes every year).

## Limitations
- **The 96% models explain; only the pregame model forecasts.** Don't quote 96% as prediction accuracy.
- **Team level only.** It doesn't say which *players* drive these stats.
- **Regular season only.** Playoff basketball (repeated opponents, tighter rotations) may behave differently.

---

# Part 2: Moneyball. Rating players, projecting 2026-27, finding trades

Part 1 asked *which stats win games*. Part 2 turns that into decisions: **how good is each player, how many games will each team win this season, who should they trade for, and who will win the title?** Data runs through **November 11, 2026**, about 12 games into the season.

![Championship odds](figures/championship_odds.png)

![Projected wins](figures/projected_wins.png)

## Key findings
1. **Boston (22.7%) and San Antonio (17.6%) are the title favorites**, followed by Detroit (11.3%). No team is above 25%, so the playoffs are wide open.
2. **Hot starts can mislead.** Philadelphia is 9-3 but projects for only 43 wins. The Lakers are 11-2 but project below Houston and OKC. The rosters don't support the early records.
3. **The best players by our rating:** Shai Gilgeous-Alexander, Victor Wembanyama and Nikola Jokić. These ratings come from a method we built, not one we borrowed.
4. **Top trade targets for contenders:** Devin Booker, Michael Porter Jr., Giannis Antetokounmpo and Brook Lopez. For example, OKC adding Booker projects as about **+4.5 wins**.

## How it works

| Step | What happens | File |
|---|---|---|
| Player data | Every player's box score for every game, 2015-16 to today, plus current rosters and the remaining schedule | [`src/fetch_players.py`](src/fetch_players.py) |
| Player rating | **RAPM with a box-score prior.** Box stats give a first guess (learned from history), and the team's actual margin with the player on the floor adjusts it. Uses this season + last season (+ the season before at half weight) and a learned **age curve** | [`src/ratings.py`](src/ratings.py) |
| Injuries | Reads the NBA's **official injury report PDFs** and sorts each injury into an expected absence (e.g. fresh ACL tear = out for the season) | [`src/injuries.py`](src/injuries.py) |
| Team strength | Each player's rating × expected minutes (blending this season and last, adjusted for availability and injuries) | [`src/project.py`](src/project.py) |
| Win projection | Every remaining game becomes a win probability from the strength gap + home court | [`src/project_current.py`](src/project_current.py) |
| Trades | For each contender, try every healthy veteran on a losing team and measure the extra wins | [`src/trades.py`](src/trades.py) |
| Title odds | Simulate the rest of the season, play-in and playoffs **20,000 times** | [`src/simulate.py`](src/simulate.py) |

## Does it work? A backtest on 9 past seasons
I replayed 2017-18 to 2025-26, stopping each one at the same point in the season as today and using **only data available at that moment**:

| Method | Average miss (wins) |
|---|---|
| **This model** | **6.0** (6.1 before tuning) |
| Average of last season and record so far | 6.7 |
| Record so far | 8.7 |
| Last season's record | 8.7 |

**Caveat:** the final settings (a 3-season window, the age curve and the regularization strength) were chosen by looking at this same backtest, so 6.0 is slightly optimistic. The untuned version scored 6.1, so the honest range is 6.0–6.1. An audit also fixed a small leak (league averages briefly used the full season), which changed the result by only 0.01.

Even with perfect knowledge of every team's true strength, luck alone would cause about 2.4 wins of error. Most of the remaining gap comes from things no model can see in advance: future injuries, trades and breakout seasons. The backtest also measured how far a team's true strength can be from our estimate (±3.3 points), and the simulation includes that uncertainty so the odds aren't overconfident.

## Limitations
- **Trades ignore salary and team fit.** The model knows how many wins a player is worth, not whether he fits a roster's needs or whether the contracts can be matched. That's why most contenders name the same few targets. **Team fit is the next planned upgrade.**
- **Game-level, not possession-level, plus-minus.** Bench players on dominant teams can come out a little high.
- **Injury return dates are estimated** from the injury description, since the official report lists status, not timelines.

## Run it yourself
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/fetch_data.py        # optional: data/raw is already included
python src/fetch_shot_zones.py  # optional: shot locations for Part 1c
python src/build_features.py
python src/build_pregame_features.py
python src/build_player_strength.py
jupyter notebook notebooks/analysis.ipynb   # Part 1, 1b-1e: models, feature importance, shot value, player model, market test

# Part 2 (data/raw is included; re-fetch to update to today)
python src/fetch_players.py && python src/fetch_ages.py && python src/fetch_injuries.py
cd src
python backtest.py          # check the method on past seasons
python project_current.py   # 2026-27 projected wins
python trades.py            # trade targets
python simulate.py          # championship odds
python plot_moneyball.py    # charts
```

**Data credit:** betting odds from [MGM Grand NBA betting data](https://www.kaggle.com/datasets/caseydurfee/mgm-grand-nba-betting-data) by Casey Durfee (CC BY-SA 4.0), saved as `data/raw/betting_odds_mgm.csv`. Game and player stats from stats.nba.com via `nba_api`.

**Tools:** Python, pandas, NumPy, SciPy, scikit-learn, matplotlib, Jupyter, nba_api, pdfplumber
