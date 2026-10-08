"""Loss or corruption of raw evidence must prevent a completed checkpoint."""
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch

from at2hjo.revision4.config import load
from at2hjo.revision4 import runner


ROOT = Path(__file__).resolve().parents[1]


def _assert_corrupted_log_refused(tmp_path, monkeypatch, corrupt, expected_message):
    torch.set_num_threads(1)
    c = load(ROOT / 'configs/revision4/smoke.json')
    c.update(horizon=2, episodes=1, hidden=[8, 8], attention=4, value_hidden=4)
    actual_verify = runner.verify_episode_logs

    def corrupt_before_verification(output, *args, **kwargs):
        corrupt(output / 'slots.jsonl')
        return actual_verify(output, *args, **kwargs)

    monkeypatch.setattr(runner, 'load', lambda _: c)
    monkeypatch.setattr(runner, 'verify_episode_logs', corrupt_before_verification)
    args = SimpleNamespace(command='train', config='in_memory', checkpoint=None,
                           output=str(tmp_path / 'run'), seed=2000, device='cpu',
                           episodes=1, algorithm=None, trust_checkpoint=False)
    with pytest.raises(RuntimeError, match=expected_message):
        runner.execute(args)
    status = json.loads((tmp_path / 'run/status.json').read_text())
    assert status['status'] == 'failed'
    assert not (tmp_path / 'run/evaluation.pt').exists()
    assert not (tmp_path / 'run/resume.pt').exists()


def test_missing_final_slot_refuses_completion_before_checkpoint(tmp_path, monkeypatch):
    def corrupt(path):
        rows = path.read_text(encoding='utf-8').splitlines(keepends=True)
        assert len(rows) == 2
        path.write_text(''.join(rows[:-1]), encoding='utf-8')

    _assert_corrupted_log_refused(tmp_path, monkeypatch, corrupt, 'raw slot log')


def test_corrupted_logged_cost_refuses_completion_before_checkpoint(tmp_path, monkeypatch):
    def corrupt(path):
        rows = [json.loads(row) for row in path.read_text().splitlines()]
        rows[-1]['cost_task'] += 1.
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

    _assert_corrupted_log_refused(tmp_path, monkeypatch, corrupt, 'Raw component audit')
