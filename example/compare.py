"""Compare matched final-state outputs. Only Python's standard library is needed."""
import argparse
import csv
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('cpu_output', type=Path)
parser.add_argument('gpu_output', type=Path)
parser.add_argument('--position-tolerance-m', type=float, default=1e-3)
parser.add_argument('--velocity-tolerance-m-s', type=float, default=1e-6)
args = parser.parse_args()
for tolerance in [args.position_tolerance_m, args.velocity_tolerance_m_s]:
    if not math.isfinite(tolerance) or tolerance <= 0:
        parser.error('Tolerances must be finite and positive.')

def read(directory):
    with (directory/'final_states.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Empty output: '+str(directory))
    result = {row['id']: row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate target IDs')
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    if not summary['validation_passed']:
        raise ValueError('Propagation validation failed')
    return result

cpu, gpu = read(args.cpu_output), read(args.gpu_output)
if cpu.keys() != gpu.keys():
    raise ValueError('Target IDs differ')
positions, velocities = [], []
for key in cpu:
    a, b = cpu[key], gpu[key]
    values = [float(row[column]) for row in (a,b) for column in row if column != 'id']
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Nonfinite output')
    if abs(float(a['epoch_mjd2000_tdb'])-float(b['epoch_mjd2000_tdb']))*86400 > 1e-4:
        raise ValueError('Final epochs differ')
    positions.append(1000*math.sqrt(sum((float(a[k])-float(b[k]))**2 for k in ['x_km','y_km','z_km'])))
    velocities.append(1000*math.sqrt(sum((float(a[k])-float(b[k]))**2 for k in ['vx_km_s','vy_km_s','vz_km_s'])))
report = {'targets':len(cpu), 'max_position_difference_m':max(positions),
    'max_velocity_difference_m_s':max(velocities),
    'passed':max(positions)<=args.position_tolerance_m and max(velocities)<=args.velocity_tolerance_m_s,
    'note':'Implementation consistency check, not comparison with a precise ephemeris.'}
print(json.dumps(report,indent=2))
raise SystemExit(0 if report['passed'] else 1)
