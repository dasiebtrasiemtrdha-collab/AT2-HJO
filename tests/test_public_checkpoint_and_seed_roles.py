"""Public entry-point regressions for checkpoint trust and validation roles."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json

import pytest
import torch

from at2hjo.revision4.config import load
from at2hjo.revision4.environment import Environment
from at2hjo.revision4.learning import Learner
from at2hjo.revision4.runner import checkpoint, execute


ROOT = Path(__file__).resolve().parents[1]


def _evaluation_checkpoint(tmp_path):
    c = load(ROOT / 'configs/revision4/smoke.json')
    c.update(horizon=1, episodes=1, hidden=[8, 8], attention=4,
             value_hidden=4, validation_seeds=[3000])
    learner = Learner(Environment(c, 2000), 'cpu', 2000)
    path = tmp_path / 'evaluation.pt'
    checkpoint(path, learner, c, 0, False)
    return path


def _args(tmp_path, path, command='evaluate', seed=3000):
    return SimpleNamespace(command=command, output=str(tmp_path / 'run'),
                           device='cpu', checkpoint=str(path), config=None,
                           algorithm=None, seed=seed, episodes=1, dataset=None,
                           trust_checkpoint=False)


def test_validation_seed_is_read_only_and_recorded_as_validation(tmp_path):
    torch.set_num_threads(1)
    args = _args(tmp_path, _evaluation_checkpoint(tmp_path))
    original_load = torch.load
    with patch('torch.load', wraps=original_load) as checked_load:
        execute(args)
    assert checked_load.call_args.kwargs['weights_only'] is True
    run = json.loads((tmp_path / 'run/run.json').read_text())
    status = json.loads((tmp_path / 'run/status.json').read_text())
    assert run['seed_role'] == 'validation'
    assert status['evaluation_unchanged'] is True
    assert status['counters']['environment_steps'] == 0


def test_resume_requires_explicit_trust_before_deserializing(tmp_path):
    args = _args(tmp_path, tmp_path / 'unread.pt', 'resume', seed=2000)
    with patch('torch.load', side_effect=AssertionError('must not deserialize')):
        with pytest.raises(ValueError, match='trust|trusted'):
            execute(args)
    assert not (tmp_path / 'run').exists()
