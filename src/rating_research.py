"""Research alternatives to the frozen game-level player rating.

These remain game-level models: teammate effects are not fully identifiable
without possession/lineup data. No production ratings are overwritten.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

from ratings import (STATS, LAMBDA, BOX_SHRINK_MIN, REPLACEMENT,
                     OLDER_SEASON_WEIGHT, design, box_rating, league_per36,
                     prev_season, season_ratings)

ARMS = ('legacy', 'joint_home', 'recent_365', 'recent_normalized')
HALF_LIFE_DAYS = 365.


def recency_weights(dates, cutoff, half_life=HALF_LIFE_DAYS):
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError('Half life must be positive')
    elapsed = (pd.Timestamp(cutoff) - pd.to_datetime(dates)).dt.total_seconds() / 86400
    if elapsed.isna().any() or (elapsed <= 0).any():
        raise ValueError('Ratings require strictly earlier games')
    return np.exp2(-elapsed / half_life)


def joint_fit(X, y, h, prior, weights, lam=LAMBDA):
    """Weighted ridge about prior, with an unpenalized home coefficient.

    Minimize sum(w * (y - X r - h b)^2) + lam * ||r-prior||^2.
    Home advantage is estimated with player effects, using the same weights.
    """
    y, h, prior, weights = [np.asarray(a, dtype=float) for a in (y, h, prior, weights)]
    if X.shape != (len(y), len(prior)) or h.shape != y.shape or weights.shape != y.shape:
        raise ValueError('Mismatched rating inputs')
    if any(not np.isfinite(a).all() for a in (y, h, prior, weights)) or not np.isfinite(X.data).all():
        raise ValueError('Nonfinite rating input')
    if lam <= 0 or not np.isfinite(lam) or (weights <= 0).any():
        raise ValueError('Positive penalty and weights required')
    root = np.sqrt(weights)
    A = sparse.hstack([X, sparse.csr_matrix(h[:, None])], format='csr').multiply(root[:, None])
    penalty = sparse.hstack([np.sqrt(lam) * sparse.eye(X.shape[1]),
                             sparse.csr_matrix((X.shape[1], 1))], format='csr')
    A = sparse.vstack([A, penalty], format='csr')
    target = np.r_[root * (y - X @ prior), np.zeros(X.shape[1])]
    solution = lsqr(A, target, atol=1e-10, btol=1e-10, iter_lim=4000)
    if solution[1] not in (0, 1, 2, 4, 5):
        raise RuntimeError(f'Rating solver did not converge: {solution[1]}')
    return prior + solution[0][:-1], float(solution[0][-1])


def weighted_box_prior(sub, weights, box_model):
    """Apply the same recency weights to box rates, league baselines and shrinkage."""
    weighted = sub.copy()
    weighted[STATS + ['MIN']] = weighted[STATS + ['MIN']].mul(weights, axis=0)
    league = league_per36(weighted)
    prior = box_rating(weighted, box_model, league)
    effective = weighted.groupby('PLAYER_ID').MIN.sum()
    return prior, effective, weighted


def rating_variants(logs, season, box_model, age_curve, ages, cutoff, impact_penalties=()):
    """All arms share training windows, box coefficients and the learned age curve."""
    past = logs[pd.to_datetime(logs.GAME_DATE) < pd.Timestamp(cutoff)].copy()
    p1, p2 = prev_season(season), prev_season(prev_season(season))
    sub = past[past.SEASON.isin([p2, p1, season])].copy()
    if sub.empty or (sub.MIN <= 0).any() or sub.duplicated(['GAME_ID', 'PLAYER_ID']).any():
        raise ValueError('Empty, duplicate or nonpositive-minute history')
    legacy = season_ratings(past, season, box_model, age_curve, ages, cutoff).set_index('PLAYER_ID')
    X, y, h, players, seasons = design(sub)
    # design orders rows by GAME_ID, irrespective of input row order.
    game_dates = sub.groupby('GAME_ID').GAME_DATE.first().sort_index()
    if len(game_dates) != len(y):
        raise ValueError('Incomplete game in rating history')
    sw = np.where(seasons == p2, OLDER_SEASON_WEIGHT, 1.)
    legacy['EFFECTIVE_MIN'] = sub.assign(WMIN=sub.MIN * sub.SEASON.map(
        {p2: OLDER_SEASON_WEIGHT, p1: 1., season: 1.})).groupby('PLAYER_ID').WMIN.sum()
    legacy['IMPACT_ADJ'] = legacy.RATING - legacy.BOX_RATING - legacy.AGE_ADJ
    outputs = {'legacy': legacy}
    prior = legacy.BOX_RATING.reindex(players).to_numpy()
    joint, home = joint_fit(X, y, h, prior, sw)
    result = legacy.copy()
    result['IMPACT_ADJ'] = pd.Series(joint - prior, index=players)
    result['RATING'] = result.BOX_RATING + result.IMPACT_ADJ + result.AGE_ADJ
    result.attrs['home_edge'] = home
    outputs['joint_home'] = result

    row_weights = recency_weights(sub.GAME_DATE, cutoff)
    recent_prior, effective, weighted = weighted_box_prior(sub, row_weights, box_model)
    prior = recent_prior.reindex(players).to_numpy()
    recent, home = joint_fit(X, y, h, prior, recency_weights(game_dates, cutoff).to_numpy())
    result = legacy.copy()
    result['BOX_RATING'] = recent_prior
    result['EFFECTIVE_MIN'] = effective
    current_min = weighted[weighted.SEASON == season].groupby('PLAYER_ID').MIN.sum()
    older_share = 1 - current_min.reindex(result.index).fillna(0) / effective.reindex(result.index)
    # Keep the existing one-year age-curve convention, changing only its evidence weights.
    result['AGE_ADJ'] = age_curve(result.AGE.to_numpy()) * older_share
    result['IMPACT_ADJ'] = pd.Series(recent - prior, index=players)
    result['RATING'] = result.BOX_RATING + result.IMPACT_ADJ + result.AGE_ADJ
    result.attrs['home_edge'] = home
    outputs['recent_365'] = result
    # Preserve total game weight while changing its distribution over time.
    # This controls global penalty strength, not each player's sample size.
    game_weights = recency_weights(game_dates, cutoff).to_numpy()
    scale = sw.sum() / game_weights.sum()
    norm_prior, effective, weighted = weighted_box_prior(sub, row_weights * scale, box_model)
    prior = norm_prior.reindex(players).to_numpy()
    normalized, home = joint_fit(X, y, h, prior, game_weights * scale)
    result = outputs['recent_365'].copy()
    result['BOX_RATING'] = norm_prior
    result['EFFECTIVE_MIN'] = effective
    result['IMPACT_ADJ'] = pd.Series(normalized - prior, index=players)
    result['RATING'] = result.BOX_RATING + result.IMPACT_ADJ + result.AGE_ADJ
    result.attrs['home_edge'] = home
    result.attrs['weight_scale'] = scale
    outputs['recent_normalized'] = result
    for penalty in impact_penalties:
        fitted, home = joint_fit(X, y, h, prior, game_weights * scale, lam=penalty)
        conservative = result.copy()
        conservative['IMPACT_ADJ'] = pd.Series(fitted-prior,index=players)
        conservative['RATING'] = conservative.BOX_RATING + conservative.IMPACT_ADJ + conservative.AGE_ADJ
        conservative.attrs['home_edge'] = home
        outputs[f'impact_{penalty:g}'] = conservative
    return outputs
