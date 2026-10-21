"""Run every original monolithic case in isolation, pytest, integration and CUDA benchmarks."""
import argparse
import datetime
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def execute(command, log, timeout):
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    row = {'command': command, 'timeout_seconds': timeout, 'log': log.name}
    start = time.monotonic()
    with log.open('w', encoding='utf-8') as stream:
        try:
            options = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, **options)
            try:
                code = process.wait(timeout=timeout)
                row.update(status='passed' if code == 0 else 'failed', exit_code=code)
            except subprocess.TimeoutExpired:
                # Stop only this test process and its descendants, never other Python jobs.
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                   stdout=stream, stderr=subprocess.STDOUT, timeout=30)
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
                row['status'] = 'timeout'
        except Exception as exc:
            row.update(status='runner_error', error=repr(exc))
    row['seconds'] = time.monotonic() - start
    if log.exists():
        messages = log.read_text(encoding='utf-8', errors='replace').count(
            'Windows fatal exception: access violation')
        if messages:
            row['windows_access_violation_messages'] = messages
    return row


def case_child(name):
    from helioslm_v5.tests import test_v5
    cases = [f for f in test_v5.TESTS if f.__name__ == name]
    assert len(cases) == 1, name
    cases[0]()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', help=argparse.SUPPRESS)
    ap.add_argument('--list-cases', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--case-timeout', type=int, default=180)
    ap.add_argument('--suite-timeout', type=int, default=1800)
    ap.add_argument('--self-check', action='store_true', help='check orchestrator only; no GPU or regression claim')
    args = ap.parse_args()
    if args.list_cases:
        from helioslm_v5.tests import test_v5
        print(json.dumps([f.__name__ for f in test_v5.TESTS]))
        return 0
    if args.case:
        case_child(args.case)
        return 0
    if min(args.case_timeout, args.suite_timeout) <= 0:
        ap.error('timeouts must be positive')
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = ROOT / 'extended_results' / stamp
    output.mkdir(parents=True)
    summary = {'status': 'running', 'jobs': [], 'cuda_checks_passed': False,
               'full_regression_passed': False, 'scope': 'single GPU; original regression fixtures mostly CPU',
               'unverified': ['multi-GPU/NCCL (one GPU, Windows)',
                              'native FP8 convergence / mid-scale training (lite synthetic acceptance only)',
                              'large-model throughput / exhaustive architecture matrix']}

    def checkpoint():
        (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    def run(name, command, timeout, scope):
        print('Running: ' + name, flush=True)
        row = execute(command, output / (name + '.log'), timeout)
        row.update(name=name, scope=scope)
        summary['jobs'].append(row)
        checkpoint()
        print(name + ': ' + row['status'], flush=True)
        return row

    try:
        if args.self_check:
            run('selfcheck_success', [sys.executable, '-c', 'print("orchestrator works")'], 10, 'harness only')
            row = run('selfcheck_expected_failure', [sys.executable, '-c', 'raise SystemExit(3)'], 10, 'expected failure')
            assert row['status'] == 'failed' and row['exit_code'] == 3
            row['status'] = 'expected_failure_observed'
            row = run('selfcheck_expected_timeout', [sys.executable, '-c', 'import time;time.sleep(30)'], 1, 'expected timeout')
            assert row['status'] == 'timeout'
            row['status'] = 'expected_timeout_observed'
            summary['status'] = 'orchestrator_selfcheck_passed'
        else:
            run('cuda_areas', [sys.executable, '-u', 'scripts/verify_cuda.py', '--out', str(output / 'cuda_report.json')],
                600, 'explicit CUDA device')
            report = output / 'cuda_report.json'
            if report.exists():
                summary['cuda_checks_passed'] = json.loads(report.read_text())['cuda_checks_passed']
            run('native_fp8_training', [sys.executable, '-u', 'scripts/train_native_fp8.py',
                                       '--steps', '10', '--out', str(output / 'native_fp8_training.json')],
                600, 'full lite synthetic training; native FP8 forward and gradient GEMMs; single GPU')
            # Discover in a short-lived child: importing test_v5 imports Torch,
            # which otherwise stays resident in the coordinator during every job.
            inventory = subprocess.check_output(
                [sys.executable, str(Path(__file__).resolve()), '--list-cases'],
                cwd=ROOT, timeout=args.case_timeout, text=True, encoding='utf-8')
            names = json.loads(inventory)
            summary['original_cases'] = names
            for i, name in enumerate(names):
                run('case_%03d_%s' % (i, name), [sys.executable, '-u', str(Path(__file__).resolve()), '--case', name],
                    args.case_timeout, 'original monolithic case; primarily CPU; isolated process')
            if importlib.util.find_spec('pytest') is None:
                summary['jobs'].append({'name': 'pytest', 'status': 'blocked_missing_pytest',
                                        'remedy': 'python -m pip install pytest (leave torch unchanged)'})
                checkpoint()
            else:
                # Keep the mid allocation in a fresh process, and report failures
                # by name even if a later native operation crashes.
                run('pytest', [sys.executable, '-u', '-m', 'pytest', 'helioslm_v5/tests/', '-vv', '-rA',
                               '--tb=short', '--junitxml=' + str(output / 'pytest.xml'),
                               '--ignore=helioslm_v5/tests/test_v5.py',
                               '--ignore=helioslm_v5/tests/test_v5_cases.py',
                               '--ignore=helioslm_v5/tests/test_v533_mid.py'],
                    args.suite_timeout, 'pytest excluding separately executed original cases and mid acceptance tests')
                run('pytest_mid', [sys.executable, '-u', '-m', 'pytest',
                                  'helioslm_v5/tests/test_v533_mid.py', '-vv', '-rA', '--tb=short',
                                  '--junitxml=' + str(output / 'pytest_mid.xml')],
                    args.suite_timeout, 'mid acceptance tests in a fresh process')
            run('integration', [sys.executable, '-u', 'integration_test_v51.py'], args.suite_timeout, 'original integration; primarily CPU')
            run('cuda_benchmark', [sys.executable, '-u', 'scripts/bench_cuda_local.py', '--out', str(output / 'cuda_benchmark.json')],
                600, 'small synchronized single-GPU measurements and FP8 operator dispatch')
            regression = [r for r in summary['jobs'] if r['name'].startswith('case_') or r['name'] in ('pytest', 'pytest_mid', 'integration')]
            summary['full_regression_passed'] = all(r['status'] == 'passed' for r in regression)
            summary['status'] = 'passed' if all(r['status'] == 'passed' for r in summary['jobs']) else 'failed_or_partial'
    except Exception as exc:
        import traceback
        summary.update(status='runner_error', error=repr(exc), traceback=traceback.format_exc())
    summary['windows_diagnostic_jobs'] = [r['name'] for r in summary['jobs']
                                           if r.get('windows_access_violation_messages', 0)]
    checkpoint()
    archive = ROOT / ('HeliosLM_Extended_Results_' + stamp + '.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in output.rglob('*'):
            if path.is_file():
                z.write(path, path.relative_to(output))
    print('\nSend this ZIP back:\n' + str(archive), flush=True)
    print('Status: ' + summary['status'], flush=True)
    return 0 if summary['status'] in ('passed', 'orchestrator_selfcheck_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
