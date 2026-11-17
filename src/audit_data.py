"""Read-only data consistency and provenance audit. Never modifies raw inputs."""
import hashlib
import datetime as dt
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from grade_predictions import score_games, wilson_interval

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/data_audit'


def audit_teams(teams, players):
    d = teams.copy()
    if d.duplicated(['GAME_ID','TEAM_ID']).any():
        raise ValueError('Duplicate team-game rows')
    counts = d.groupby('GAME_ID').size()
    valid = d.GAME_ID.map(counts)==2
    d['DERIVED_MARGIN'] = (2*d.PTS-d.groupby('GAME_ID').PTS.transform('sum')).where(valid)
    bad_margin = d[valid & ~np.isclose(d.PLUS_MINUS,d.DERIVED_MARGIN)].copy()
    home_count = d.assign(HOME=d.MATCHUP.str.contains('vs.',regex=False)).groupby('GAME_ID').HOME.sum()
    bad_home = home_count[home_count!=1].index
    columns = ['PTS','FGM','FGA','FG3M','FTM']
    sums = players.groupby(['GAME_ID','TEAM_ID'])[columns].sum()
    m = d.merge(sums.add_suffix('_PLAYERS'),on=['GAME_ID','TEAM_ID'],how='left',validate='one_to_one')
    mismatches = {c:int((m[c]!=m[c+'_PLAYERS']).sum()) for c in columns}
    fga = m[m.FGA!=m.FGA_PLAYERS][['GAME_ID','GAME_DATE','MATCHUP','FGA','FGA_PLAYERS']]
    stats = {'team_rows':len(d),'games':int(d.GAME_ID.nunique()),
        'non_two_team_games':int((counts!=2).sum()),
        'score_formula_failures':int((d.PTS!=2*d.FGM+d.FG3M+d.FTM).sum()),
        'made_exceeds_attempted':int(((d.FGM>d.FGA)|(d.FG3M>d.FG3A)|(d.FTM>d.FTA)).sum()),
        'rebound_formula_failures':int((d.REB!=d.OREB+d.DREB).sum()),
        'winner_label_failures':int((valid & ((d.WL=='W')!=(d.DERIVED_MARGIN>0))).sum()),
        'margin_disagreement_rows':len(bad_margin),'margin_disagreement_games':int(bad_margin.GAME_ID.nunique()),
        'ambiguous_home_away_games':len(bad_home),'ambiguous_home_away_ids':bad_home.tolist(),
        'player_sum_disagreements':mismatches,
        'missing_player_team_games':int(m.PTS_PLAYERS.isna().sum()),
        'seasons':d.groupby('SEASON').GAME_ID.nunique().astype(int).to_dict()}
    return stats,bad_margin,fga,d


def main():
    hashes = {}
    def read(relative):
        content=(ROOT/relative).read_bytes()
        hashes[relative]=hashlib.sha256(content).hexdigest()
        return pd.read_csv(io.BytesIO(content),compression='gzip' if relative.endswith('.gz') else None,
                           dtype={'GAME_ID':str,'gameId':str})
    t=read('data/raw/team_game_logs.csv')
    p=read('data/raw/player_game_logs.csv.gz')
    ct=read('data/raw/team_game_logs_current.csv')
    cp=read('data/raw/player_game_logs_current.csv.gz')
    schedule=read('data/raw/schedule_current.csv')
    predictions=read('predictions/games.csv')
    odds=read('data/raw/betting_odds_mgm.csv')
    OUT.mkdir(parents=True,exist_ok=True)
    report={'audit_started_at_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    for name,teams,players in [('historical',t,p),('current',ct,cp)]:
        stats,margin,fga,checked=audit_teams(teams,players)
        stats['player_rows']=len(players)
        stats['duplicate_player_game_rows']=int(players.duplicated(['GAME_ID','PLAYER_ID']).sum())
        stats['player_score_formula_failures']=int((players.PTS!=2*players.FGM+players.FG3M+players.FTM).sum())
        report[name]=stats
        margin.to_csv(OUT/f'{name}_margin_disagreements.csv',index=False)
        fga.to_csv(OUT/f'{name}_fga_disagreements.csv',index=False)
        checked[['GAME_ID','GAME_DATE','SEASON','TEAM_ID','TEAM_ABBREVIATION','PTS','PLUS_MINUS','DERIVED_MARGIN']].to_csv(
            OUT/f'{name}_checked_results.csv',index=False)
    compared=[]
    for side in ['home','away']:
        rows=schedule[['gameId',f'{side}Team_teamId',f'{side}Team_score','gameStatus']].rename(
            columns={'gameId':'GAME_ID',f'{side}Team_teamId':'TEAM_ID',f'{side}Team_score':'SCHEDULE_PTS'})
        m=ct.merge(rows,on=['GAME_ID','TEAM_ID'],how='left',validate='one_to_one')
        m=m[m.gameStatus==3]
        compared.append({'side':side,'matched_completed_team_rows':len(m),'score_disagreements':int((m.PTS!=m.SCHEDULE_PTS).sum())})
    report['current_schedule']=compared
    graded=score_games(predictions,ct)
    graded.to_csv(OUT/'prediction_grading_preview.csv',index=False)
    done=graded[graded.HOME_WIN.notna()]
    report['live_predictions']={'rows':len(graded),'graded':len(done),
        'invalid_predictions':int((graded.GRADE_STATUS=='invalid_prediction').sum()),
        'correct':int(done.CORRECT.sum()),
        'wilson_95':list(wilson_interval(int(done.CORRECT.sum()),len(done))) if len(done) else None,
        'timestamp_provenance':'Self-recorded timestamps plus local Git history; remote publication not independently verified.'}
    report['odds']={'rows':len(odds),'duplicate_source_ids':int(odds.game_id.duplicated().sum()),
        'has_verified_quote_observation_timestamps':False}
    report['source_assurance']={'fetch_code':'nba_api calling stats.nba.com; not independent proof of saved CSV provenance',
        'remote_verification':'This script does not verify GitHub HEAD or independently fetch official results. See the dated narrative report for external spot checks.',
        'scope':'Local working copy, including uncommitted changes and previously added research files.'}
    report['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    report['snapshot_sha256']=hashes
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
