"""Local timestamped odds/availability snapshots. No bookmaker connection or bets.

Run: python src/pregame_snapshots.py --input payload.json
Observation time is supplied by the data source; receipt time is recorded here.
This records provenance claims, it does not independently authenticate a source.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def utc(value):
    t = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError('Timezone is required')
    return t.astimezone(dt.timezone.utc)


def validate(payload, received):
    received = utc(received)
    observed, tip = utc(payload['observed_at']), utc(payload['tip_utc'])
    if not observed <= received < tip:
        raise ValueError('Observation must precede receipt, and receipt must precede tip-off')
    if not isinstance(payload['game_id'], str) or not payload['game_id'].isdigit():
        raise ValueError('game_id must be a digit string')
    if not isinstance(payload['source'], str) or not payload['source'].strip():
        raise ValueError('A source is required')
    if payload['kind'] == 'odds':
        if (received - observed).total_seconds() > 300:
            raise ValueError('Odds older than five minutes')
        if payload['market'] != 'moneyline' or not payload['bookmaker'].strip():
            raise ValueError('Only identified bookmaker moneyline quotes supported')
        for k in ['home_decimal', 'away_decimal']:
            v = payload[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 1:
                raise ValueError('Invalid decimal odds')
    elif payload['kind'] == 'availability':
        if (received - observed).total_seconds() > 21600:
            raise ValueError('Availability snapshot older than six hours')
        if not isinstance(payload['team_id'], int) or isinstance(payload['team_id'], bool):
            raise ValueError('team_id required')
        players = payload['players']
        if not players or len({p['player_id'] for p in players}) != len(players):
            raise ValueError('Empty or duplicate availability list')
        for p in players:
            if not isinstance(p['player_id'], int) or isinstance(p['player_id'], bool):
                raise ValueError('Player IDs must be integers')
            if p['status'] not in {'Available', 'Probable', 'Questionable', 'Doubtful', 'Out'}:
                raise ValueError('Unknown availability status')
            cap = p.get('minute_cap', 40)
            if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or not 0 <= cap <= 48:
                raise ValueError('Invalid minute cap')
    else:
        raise ValueError('Unsupported snapshot kind')
    return payload


def record(payload, directory, now=None):
    received = (now or dt.datetime.now(dt.timezone.utc)).isoformat()
    validate(payload, received)
    content = {'received_at': received, 'payload': payload}
    encoded = json.dumps(content, sort_keys=True, allow_nan=False, ensure_ascii=False).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{digest}.json'
    # Exclusive creation: an already recorded snapshot is never overwritten.
    with path.open('x') as f:
        json.dump(content, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/snapshots')
    args = parser.parse_args()
    print(record(json.loads(args.input.read_text()), args.output))


if __name__ == '__main__':
    main()
