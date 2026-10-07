"""Short standalone build checks using the ENVISAT example (not paper experiments)."""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get('BUILD_DIR', ROOT/'build')).resolve()
OUT = ROOT/'output'
OUT.mkdir(exist_ok=True)
WORK = Path(tempfile.mkdtemp(prefix='validation-', dir=OUT))
ENV = dict(os.environ, OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE', OMP_WAIT_POLICY='PASSIVE')
PATH_KEYS = {'file','spaceWeatherFile','eopFile','ephemeris','orientations','constants'}

def resolve(value, base, key=''):
    if isinstance(value, str) and key in PATH_KEYS:
        return str((base/value).resolve())
    if isinstance(value, dict):
        return {k:resolve(v,base,k) for k,v in value.items()}
    if isinstance(value, list):
        return [resolve(v,base,key) for v in value]
    return value

def config(profile):
    p = ROOT/f'example/config/envisat_{profile}.json'
    return resolve(json.loads(p.read_text()), p.parent)

def save(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n')

def launch(backend, cfg, samples, output, expect_success=True):
    # Run outside the project directory to verify configuration path handling.
    process = subprocess.run([str(BUILD/f'bin/parahpop_{backend}'),str(cfg),str(samples),str(output)],
        cwd=tempfile.gettempdir(), env=ENV, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (WORK/f'{output.name}-{backend}-{expect_success}.log').write_text(process.stdout)
    if expect_success != (process.returncode == 0):
        raise RuntimeError(f'{output.name}/{backend}: exit {process.returncode}\n{process.stdout[-4000:]}')
    if expect_success:
        assert json.loads((output/'summary.json').read_text())['validation_passed']
    return process.returncode

def read_states(path):
    with (path/'final_states.csv').open() as stream:
        return list(csv.DictReader(stream))

def pair(name, cfg, samples):
    cfg_path, input_path = WORK/f'{name}-config.json', WORK/f'{name}-input.json'
    save(cfg_path,cfg); save(input_path,samples)
    for backend in ('cpu','gpu'):
        launch(backend,cfg_path,input_path,WORK/f'{name}-{backend}')
    check = subprocess.run(['python3',str(ROOT/'example/compare.py'),str(WORK/f'{name}-cpu'),str(WORK/f'{name}-gpu')],
        env=ENV, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if check.returncode: raise RuntimeError(check.stdout)
    result = json.loads(check.stdout)
    print(name, result['max_position_difference_m'], 'm', flush=True)
    return result

samples = json.loads((ROOT/'example/input/envisat.json').read_text())
report = {'cases':{}, 'work_directory':str(WORK), 'note':'Short interface/build checks, not the paper accuracy/performance dataset.'}
report['cases']['full_600s'] = pair('full_600s',config('full'),samples)
report['cases']['central_600s'] = pair('central_600s',config('central'),samples)
endpoint = config('full'); endpoint['run']['duration_seconds'] = 65.0
report['cases']['full_65s_endpoint'] = pair('full_65s_endpoint',endpoint,samples)
short = config('full'); short['run']['duration_seconds'] = 5.0
report['cases']['full_5s_endpoint'] = pair('full_5s_endpoint',short,samples)

# Check target indexing with different epochs/parameters and >1 CPU thread.
batch = json.loads(json.dumps(samples)); batch['targets'] = []
for i in range(4):
    item = dict(samples['targets'][0])
    item['id'] = i+1; item['epoch_mjd2000_tdb'] += i/86400
    item['mass_kg'] += i*10
    batch['targets'].append(item)
batch_cfg = config('full'); batch_cfg['run']['duration_seconds'] = 65.0
batch_cfg['execution']['maxThreads'] = 2
report['cases']['batch_4'] = pair('batch_4',batch_cfg,batch)

# Cross a 32-thread model block boundary with a partial final block.
wide = json.loads(json.dumps(samples)); wide['targets'] = []
for i in range(33):
    item = dict(samples['targets'][0])
    item['id'] = i+1; item['epoch_mjd2000_tdb'] += i/86400
    item['mass_kg'] += i*10
    wide['targets'].append(item)
report['cases']['batch_33'] = pair('batch_33',batch_cfg,wide)

# A central-only case also has invariants independent of CPU/GPU agreement.
def invariants(state):
    r,v = state[:3],state[3:]
    energy = sum(x*x for x in v)/2 - 398600.4415/math.sqrt(sum(x*x for x in r))
    angular = [r[1]*v[2]-r[2]*v[1],r[2]*v[0]-r[0]*v[2],r[0]*v[1]-r[1]*v[0]]
    return energy,angular
initial_energy, initial_h = invariants(samples['targets'][0]['state_km_km_s'])
columns = ['x_km','y_km','z_km','vx_km_s','vy_km_s','vz_km_s']
report['central_invariants'] = {}
for backend in ('cpu','gpu'):
    row = read_states(WORK/f'central_600s-{backend}')[0]
    energy,h = invariants([float(row[k]) for k in columns])
    de = abs(energy-initial_energy)/abs(initial_energy)
    dh = math.sqrt(sum((a-b)**2 for a,b in zip(h,initial_h)))/math.sqrt(sum(a*a for a in initial_h))
    if de>1e-9 or dh>1e-9: raise RuntimeError(f'Central invariants failed: {backend}, {de}, {dh}')
    report['central_invariants'][backend] = {'relative_energy_change':de,'relative_angular_momentum_change':dh}

bad = config('full'); bad['execution']['integration']['maxNumSteps'] = 1
save(WORK/'insufficient-steps.json',bad)
for backend in ('cpu','gpu'):
    output = WORK/f'insufficient-{backend}'
    launch(backend,WORK/'insufficient-steps.json',ROOT/'example/input/envisat.json',output,False)
    assert not (output/'summary.json').exists()
    launch(backend,ROOT/'example/config/envisat_full.json',ROOT/'example/input/envisat.json',WORK/f'full_600s-{backend}',False)
report['failure_checks'] = {'insufficient_steps_rejected':True,'existing_outputs_preserved':True}
for field in ('absTol', 'maxStepSize'):
    unsupported = config('full')
    unsupported['execution']['integration'][field] = 1.0
    path = WORK/f'unsupported-{field}.json'; save(path,unsupported)
    launch('cpu',path,ROOT/'example/input/envisat.json',WORK/f'unsupported-{field}',False)
for value in (0,-1):
    unsupported = config('full'); unsupported['execution']['maxThreads'] = value
    path = WORK/f'bad-threads-{value}.json'; save(path,unsupported)
    launch('cpu',path,ROOT/'example/input/envisat.json',WORK/f'bad-threads-{value}',False)
invalid = json.loads(json.dumps(samples)); invalid['targets'][0]['epoch_mjd2000_tdb'] += 365.0
save(WORK/'outside-coverage.json',invalid)
launch('cpu',ROOT/'example/config/envisat_full.json',WORK/'outside-coverage.json',WORK/'outside-coverage',False)
report['failure_checks'].update(removed_options_rejected=True,invalid_thread_counts_rejected=True,
                              outside_orientation_coverage_rejected=True)
report['passed'] = True
save(WORK/'validation.json',report)
print('PASS:',WORK/'validation.json',flush=True)
