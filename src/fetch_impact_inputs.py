"""Archive dated player impact inputs; never substitute missing defensive data."""
import argparse
import datetime as dt
import json
from pathlib import Path
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashptdefend

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season', default='2026-27')
    args = parser.parse_args()
    now = dt.datetime.now(dt.timezone.utc)
    dest = ROOT/'data/raw/player_impact'/now.strftime('%Y%m%dT%H%M%S%fZ')
    dest.mkdir(parents=True)
    requests = {
        'base_per36': lambda: leaguedashplayerstats.LeagueDashPlayerStats(
            season=args.season, measure_type_detailed_defense='Base',
            per_mode_detailed='Per36', timeout=15),
        'advanced': lambda: leaguedashplayerstats.LeagueDashPlayerStats(
            season=args.season, measure_type_detailed_defense='Advanced', timeout=15),
        'misc': lambda: leaguedashplayerstats.LeagueDashPlayerStats(
            season=args.season, measure_type_detailed_defense='Misc',
            per_mode_detailed='Per36', timeout=15),
        'defense': lambda: leaguedashptdefend.LeagueDashPtDefend(
            season=args.season, defense_category='Overall', timeout=15),
        'defense_three': lambda: leaguedashptdefend.LeagueDashPtDefend(
            season=args.season, defense_category='3 Pointers', timeout=15),
        'defense_under_6ft': lambda: leaguedashptdefend.LeagueDashPtDefend(
            season=args.season, defense_category='Less Than 6Ft', timeout=15),
    }
    manifest = dict(season=args.season, schema_version=2, started_at=now.isoformat(), results={},
        units={'base_per36':'Per36','misc':'Per36','advanced':'Totals','defense_three':'Totals','defense_under_6ft':'Totals'},
        limitation='Season-to-date snapshots, not historical pregame snapshots. Defensive categories and field coverage must be checked before scoring.')
    for name, fetch in requests.items():
        try:
            frames = fetch().get_data_frames()
            for i, frame in enumerate(frames):
                frame.to_csv(dest/f'{name}_{i}.csv', index=False)
            manifest['results'][name] = {'success':True, 'received_at':dt.datetime.now(dt.timezone.utc).isoformat(),
                'columns':[list(f.columns) for f in frames]}
        except Exception as exc:
            manifest['results'][name] = {'success':False, 'error':str(exc)}
    (dest/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(dest)
    if not all(r['success'] for r in manifest['results'].values()):
        raise SystemExit('Incomplete input collection; no impact score produced.')


if __name__=='__main__':
    main()
