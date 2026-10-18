"""CPU-only tests for honest CUDA-report status and explicit CPU smoke mode."""
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def test_no_gpu_is_blocked_not_a_pass(tmp_path):
    path = tmp_path / 'blocked.json'
    result = subprocess.run([sys.executable, str(_REPO / 'scripts/verify_cuda.py'),
                             '--out', str(path)],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads(path.read_text())
    assert report['status'] == 'blocked_no_cuda'
    assert report['areas'] == []
    assert report['cuda_checks_passed'] is False


def test_cpu_smoke_does_not_claim_cuda(tmp_path):
    path = tmp_path / 'cpu.json'
    result = subprocess.run([sys.executable, str(_REPO / 'scripts/verify_cuda.py'),
                            '--cpu-smoke', '--out', str(path)],
                           capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr + path.read_text()
    report = json.loads(path.read_text())
    assert report['status'] == 'cpu_smoke_passed'
    assert report['cuda_checks_passed'] is False
    assert report['execution_device'] == 'cpu'
    assert len(report['areas']) == 12
    assert all(row['device'] == 'cpu' and row['status'] == 'passed'
               for row in report['areas'])


def test_initialization_failure_is_recorded(tmp_path, monkeypatch):
    sys.path.insert(0, str(_REPO))
    from scripts import verify_cuda
    path = tmp_path / 'init-failed.json'
    monkeypatch.setattr(sys, 'argv', ['verify_cuda.py', '--out', str(path)])
    monkeypatch.setattr(verify_cuda.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(verify_cuda.torch.cuda, 'device_count', lambda: 1)

    def fail(device):
        raise RuntimeError('test initialization failure')

    monkeypatch.setattr(verify_cuda.torch.cuda, 'set_device', fail)
    assert verify_cuda.main() == 1
    report = json.loads(path.read_text())
    assert report['status'] == 'failed_cuda_initialization'
    assert report['cuda_checks_passed'] is False and report['areas'] == []
    assert 'test initialization failure' in report['traceback']


def test_source_revision_without_git(tmp_path, monkeypatch):
    sys.path.insert(0, str(_REPO))
    from scripts import verify_cuda
    monkeypatch.setattr(verify_cuda, '_REPO', tmp_path)

    def no_git(*args, **kwargs):
        raise FileNotFoundError('git unavailable')

    monkeypatch.setattr(verify_cuda.subprocess, 'check_output', no_git)
    assert verify_cuda.source_revision() == 'unversioned'
    (tmp_path / 'SOURCE_REVISION.txt').write_text('a0f0933\n')
    assert verify_cuda.source_revision() == 'a0f0933'
