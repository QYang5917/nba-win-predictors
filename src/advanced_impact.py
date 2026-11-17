"""Validated integration for externally trained advanced-impact models.

No hand-picked ranking weights. Training data must contain an independently
constructed later-period impact target and dated source snapshots.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

FEATURES = ['BASE_RATING', 'TS_PCT', 'USG_PCT', 'AST_PCT', 'TM_TOV_PCT',
            'OREB_PCT', 'DREB_PCT', 'OFF_RATING', 'DEF_RATING',
            'PTS_PAINT', 'PTS_OFF_TOV', 'THREE_PTS_PER36',
            'RIM_DEF_DIFF', 'THREE_DEF_DIFF']


def prepare(base, advanced, as_of):
    """Canonical advanced table: one player, dated snapshot, consistent units.

    PTS_PAINT / PTS_OFF_TOV must be per-36 individual scoring, not team totals.
    Defensive differences are proportions vs shooter baseline, lower is better.
    RIM_DEF_DIFF here means under-six-foot tracking, not the entire painted area.
    OFF/DEF_RATING are on-court team rates; they are contextual inputs.
    """
    required = {'PLAYER_ID', 'OBSERVED_AT'} | (set(FEATURES) - {'BASE_RATING'})
    missing = required - set(advanced.columns)
    if missing:
        raise ValueError('Missing advanced impact inputs: ' + ', '.join(sorted(missing)))
    if advanced.PLAYER_ID.duplicated().any() or base.PLAYER_ID.duplicated().any():
        raise ValueError('Duplicate player input')
    observed = pd.to_datetime(advanced.OBSERVED_AT, utc=True, errors='coerce')
    if observed.isna().any() or (observed > pd.to_datetime(as_of, utc=True)).any():
        raise ValueError('Undated or future advanced snapshot')
    # Missing values remain unscored; finite but nonsensical units are rejected.
    fractions = ['USG_PCT','AST_PCT','TM_TOV_PCT','OREB_PCT','DREB_PCT']
    for c in fractions:
        if ((advanced[c].dropna()<0) | (advanced[c].dropna()>1)).any():
            raise ValueError(f'{c} must use proportions, not percent points')
    for c in ['RIM_DEF_DIFF','THREE_DEF_DIFF']:
        if (advanced[c].dropna().abs()>1).any():
            raise ValueError(f'Invalid defensive percentage difference: {c}')
    d = base[['PLAYER_ID','RATING']].rename(columns={'RATING':'BASE_RATING'}).merge(
        advanced[list(required)], on='PLAYER_ID', how='left', validate='one_to_one')
    d['ADVANCED_COMPLETE'] = np.isfinite(d[FEATURES]).all(axis=1)
    return d


def fit_model(training, before):
    """Fit only on completed historical forward targets; not desired player ranks.

    TARGET_IMPACT must be in points per 100 possessions, with TARGET_END before
    the prediction time. Snapshot time must precede TARGET_START.
    Caller is responsible for defensible construction/attribution of the target.
    """
    times = {k:pd.to_datetime(training[k],utc=True,errors='raise')
             for k in ['OBSERVED_AT','TARGET_START','TARGET_END']}
    if not ((times['OBSERVED_AT'] < times['TARGET_START']) &
            (times['TARGET_START'] <= times['TARGET_END']) &
            (times['TARGET_END'] < pd.to_datetime(before,utc=True))).all():
        raise ValueError('Impact training target overlaps snapshot or prediction time')
    if len(training) < 100 or not np.isfinite(training[FEATURES+['TARGET_IMPACT']]).all().all():
        raise ValueError('Insufficient or incomplete impact training history')
    model = make_pipeline(StandardScaler(), Ridge(alpha=30.))
    model.fit(training[FEATURES],training.TARGET_IMPACT)
    model.impact_trained_before_ = pd.to_datetime(before,utc=True)
    return model


def score(base, advanced, model, as_of):
    d = prepare(base,advanced,as_of)
    if not hasattr(model,'impact_trained_before_') or model.impact_trained_before_ > pd.to_datetime(as_of,utc=True):
        raise ValueError('Missing or future model training provenance')
    d['ADVANCED_IMPACT'] = np.nan
    good = d.ADVANCED_COMPLETE
    if good.any():
        d.loc[good,'ADVANCED_IMPACT'] = model.predict(d.loc[good,FEATURES])
    return d[['PLAYER_ID','ADVANCED_COMPLETE','ADVANCED_IMPACT']]
