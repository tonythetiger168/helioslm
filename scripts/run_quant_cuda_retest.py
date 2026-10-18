"""Short local quantization retest; no automatic installation or upload."""
import datetime
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = ROOT / 'quant_retest_results' / stamp
    output.mkdir(parents=True)
    summary = {'status': 'running', 'jobs': [], 'cuda_comparison_passed': False,
               'source_sha256': {}, 'scope': 'quantized Linear only; speedup pending measured results'}
    sources = ['helioslm_v5/src/quantization/standard_quant.py',
               'helioslm_v5/tests/test_quant_cuda_tiles.py',
               'scripts/bench_quant_cuda_compare.py', 'scripts/run_quant_cuda_retest.py',
               'baseline_standard_quant.py']
    for name in sources:
        if (ROOT / name).is_file():
            summary['source_sha256'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    jobs = [('quant_tests', [sys.executable, '-m', 'pytest', '-q', '-rA',
                            'helioslm_v5/tests/test_quant_cuda_tiles.py',
                            'helioslm_v5/tests/case_quant.py',
                            'helioslm_v5/tests/case_quant_fused.py',
                            'helioslm_v5/tests/test_quant_calib.py'], 600),
            ('cuda_comparison', [sys.executable, '-u', 'scripts/bench_quant_cuda_compare.py',
                                 '--baseline', str(ROOT / 'baseline_standard_quant.py'),
                                 '--out', str(output / 'cuda_comparison.json')], 900)]
    try:
        for name, command, timeout in jobs:
            print('Running: ' + name, flush=True)
            row = {'name': name, 'timeout_seconds': timeout}
            if name == 'quant_tests' and importlib.util.find_spec('pytest') is None:
                row['status'] = 'blocked_missing_pytest'
            else:
                try:
                    with (output / (name + '.log')).open('w', encoding='utf-8') as stream:
                        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                                stderr=subprocess.STDOUT, timeout=timeout)
                    row.update(exit_code=result.returncode, status='passed' if result.returncode == 0 else 'failed')
                except subprocess.TimeoutExpired:
                    row['status'] = 'timeout'
            summary['jobs'].append(row)
            print(name + ': ' + row['status'], flush=True)
            (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        report = output / 'cuda_comparison.json'
        if report.exists():
            summary['cuda_comparison_passed'] = json.loads(report.read_text())['cuda_benchmark_passed']
        summary['status'] = 'passed' if summary['cuda_comparison_passed'] and all(j['status'] == 'passed' for j in summary['jobs']) else 'failed_or_partial'
    except Exception:
        summary.update(status='runner_error', traceback=traceback.format_exc())
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    archive = ROOT / ('HeliosLM_Quant_Retest_' + stamp + '.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in output.glob('*'):
            if p.is_file():
                z.write(p, p.name)
        # Carry exact tested sources for provenance and reproducibility.
        for name in sources:
            if (ROOT / name).is_file():
                z.write(ROOT / name, 'tested_source/' + name)
    print('Status: ' + summary['status'] + '\nSend this ZIP back:\n' + str(archive), flush=True)
    return 0 if summary['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
