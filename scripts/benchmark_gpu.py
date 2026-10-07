"""Compare graph revisions with a fixed ENVISAT workload; final states only.

Run once with --phase before, modify/build, then run with --phase after.
The workload and hashes are saved under the specified work directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--phase', required=True, choices=['before', 'after'])
parser.add_argument('--work', required=True, type=Path)
parser.add_argument('--repeats', type=int, default=3)
args = parser.parse_args()
work = args.work.resolve(); work.mkdir(parents=True, exist_ok=True)
if args.repeats < 1: parser.error('--repeats must be positive')
env = dict(os.environ, OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE', OMP_WAIT_POLICY='PASSIVE')

def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def hardware():
    p = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used',
                        '--format=csv,noheader'], capture_output=True, text=True)
    return p.stdout.strip() or p.stderr.strip()

if not (work/'config.json').exists():
    if args.phase != 'before': raise RuntimeError('The before workload must exist first.')
    cfg_path = ROOT/'example/config/envisat_full.json'
    keys = {'file','constants','ephemeris','orientations','spaceWeatherFile','eopFile'}
    def resolve(x,key=''):
        if isinstance(x,str) and key in keys: return str((cfg_path.parent/x).resolve())
        if isinstance(x,dict): return {k:resolve(v,k) for k,v in x.items()}
        if isinstance(x,list): return [resolve(v,key) for v in x]
        return x
    cfg = resolve(json.loads(cfg_path.read_text()))
    cfg['run']['duration_seconds'] = 86400.0
    cfg['execution'] = {'maxThreads':1, 'integration':{'initialStepSize':60.0, 'maxNumSteps':2000}}
    sample = json.loads((ROOT/'example/input/envisat.json').read_text())['targets'][0]
    save(work/'targets.json', {'targets':[dict(sample, id=i+1) for i in range(10000)]})
    save(work/'config.json', cfg)
    cfg['run']['duration_seconds'] = 600.0
    save(work/'warmup.json', cfg)

phase = work/args.phase; phase.mkdir(exist_ok=True)
if (phase/'timings.json').exists(): raise RuntimeError('Phase already measured; use its saved results.')
binary = ROOT/'build/bin/parahpop_gpu'
metadata = {'phase':args.phase, 'repeats':args.repeats, 'targets':10000, 'duration_seconds':86400,
            'step_seconds':60, 'output':'final states only',
            'hashes':{str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p.name):sha(p)
                      for p in [binary,ROOT/'build/lib/libparahpop_engine.so',work/'config.json',work/'targets.json']},
            'timing_scope':'propagation_seconds includes model/cache allocation, transfers, graph construction, replay and cleanup; excludes environment file loading and output serialization.',
            'hardware_start':hardware(), 'runs':[]}
save(phase/'metadata.json',metadata)
for name in ['warmup']+[f'run-{i+1}' for i in range(args.repeats)]:
    cfg = work/('warmup.json' if name=='warmup' else 'config.json')
    out = phase/name
    if (out/'summary.json').exists(): raise RuntimeError(f'Output already exists: {out}')
    print(f'{args.phase} {name}: starting',flush=True)
    start = time.perf_counter()
    p = subprocess.run([str(binary),str(cfg),str(work/'targets.json'),str(out)],env=env,
                       capture_output=True,text=True,timeout=3600)
    elapsed = time.perf_counter()-start
    (phase/f'{name}.log').write_text(p.stdout+p.stderr,encoding='utf-8')
    if p.returncode: raise RuntimeError(p.stdout+p.stderr)
    summary = json.loads((out/'summary.json').read_text())
    if not summary['validation_passed'] or summary['targets']!=10000: raise RuntimeError(summary)
    row = dict(name=name, wall_seconds=elapsed, **summary, hardware_end=hardware())
    save(phase/f'{name}-timing.json',row)
    if name!='warmup': metadata['runs'].append(row)
    save(phase/'metadata.json',metadata)
    print(f'{args.phase} {name}: propagation={summary["propagation_seconds"]:.6f}s, wall={elapsed:.6f}s',flush=True)
metadata['propagation_median_seconds']=statistics.median(r['propagation_seconds'] for r in metadata['runs'])
metadata['wall_median_seconds']=statistics.median(r['wall_seconds'] for r in metadata['runs'])
save(phase/'timings.json',metadata)
print(json.dumps({k:metadata[k] for k in ['phase','propagation_median_seconds','wall_median_seconds']}),flush=True)
