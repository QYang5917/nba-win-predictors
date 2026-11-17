"""Translate archived NBA endpoint responses to the advanced-impact schema."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def normalize(snapshot):
    snapshot = Path(snapshot)
    manifest = json.loads((snapshot/'manifest.json').read_text())
    if manifest.get('schema_version') != 2:
        raise ValueError('Version 2 units/provenance manifest required; re-fetch older snapshots')
    kinds = ['advanced','misc','base_per36','defense_three','defense_under_6ft']
    observed = []
    def read(kind, columns, identity='PLAYER_ID'):
        meta = manifest['results'].get(kind,{})
        if not meta.get('success') or not meta.get('received_at'):
            raise ValueError(f'Incomplete source: {kind}')
        observed.append(pd.to_datetime(meta['received_at'],utc=True,errors='raise'))
        d = pd.read_csv(snapshot/f'{kind}_0.csv')
        if not set(columns+[identity]) <= set(d.columns):
            raise ValueError(f'Unexpected {kind} response fields')
        d = d[[identity]+columns].rename(columns={identity:'PLAYER_ID'})
        if d.PLAYER_ID.duplicated().any():
            raise ValueError(f'Ambiguous player/team rows in {kind}; do not average percentages')
        return d
    if manifest['units'].get('misc')!='Per36' or manifest['units'].get('base_per36')!='Per36':
        raise ValueError('Per36 scoring inputs required')
    # NBA's on-court team ratings are contextual variables, not personal ratings.
    out = read('advanced',['TS_PCT','USG_PCT','AST_PCT','TM_TOV_PCT','OREB_PCT','DREB_PCT','OFF_RATING','DEF_RATING'])
    out = out.merge(read('misc',['PTS_PAINT','PTS_OFF_TOV']),on='PLAYER_ID',how='left',validate='one_to_one')
    threes = read('base_per36',['FG3M'])
    threes['THREE_PTS_PER36'] = 3*threes.pop('FG3M')
    out = out.merge(threes,on='PLAYER_ID',how='left',validate='one_to_one')
    for kind, prefix in [('defense_three','THREE'),('defense_under_6ft','RIM')]:
        d = read(kind,['D_FGM','D_FGA','NORMAL_FG_PCT'],'CLOSE_DEF_PERSON_ID')
        if (d.D_FGA < d.D_FGM).any() or (d[['D_FGA','D_FGM']]<0).any().any():
            raise ValueError('Invalid tracked shot counts')
        d[prefix+'_DEF_DIFF'] = d.D_FGM/d.D_FGA.replace(0,np.nan)-d.NORMAL_FG_PCT
        d = d.rename(columns={'D_FGA':prefix+'_DEF_ATTEMPTS'})
        out = out.merge(d[['PLAYER_ID',prefix+'_DEF_DIFF',prefix+'_DEF_ATTEMPTS']],on='PLAYER_ID',how='left',validate='one_to_one')
    out['OBSERVED_AT'] = max(observed).isoformat()
    out['SEASON'] = manifest['season']
    return out


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',type=Path)
    args = parser.parse_args()
    result = normalize(args.snapshot)
    result.to_csv(args.snapshot/'normalized.csv',index=False)
    print(f'{len(result)} players; missing coverage preserved as NaN')
