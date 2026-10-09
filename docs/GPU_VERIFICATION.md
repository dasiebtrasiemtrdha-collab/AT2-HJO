# Windows / CUDA verification for AT2-HJO 4.0.2

This verification covers short CPU/CUDA training, evaluation, and recovery runs. Full-scale training and multi-seed performance evaluation are outside its scope.

The Windows checks used the 4.0.2 source distribution and published 34-ST CPU example, with the supplied wheel installed in an independent environment. Before testing, the input manifests validated 187 source-distribution entries and 15 model entries.

Runtime: Python 3.12.14, NumPy 2.2.6, PyTorch 2.7.1+cu126, CUDA 12.6, SciPy 1.15.3, and one NVIDIA GeForce RTX 4060. Selected device: `cuda:0`.

Source hash: `397d12fc0031f6202aa7547c5b67bd2b5e5a678c0c06915df8c95426cb9aaf50`.

## Short GPU runs

| Run | Seed | Slots this invocation | Cumulative training slots | Critic/actor/target | DQN/TR | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| provided_cpu_model_eval | 4000 | 17 | 34 | 27/13/13 | 4/5 | 2.24 |
| train | 2000 | 34 | 34 | 27/13/13 | 4/5 | 25.64 |
| resume | 2000 | 17 | 51 | 44/22/22 | 6/7 | 15.01 |
| eval | 4000 | 17 | 51 | 44/22/22 | 6/7 | 1.64 |

The supplied CPU evaluation checkpoint loaded 57 tensors onto CUDA and completed one 17-slot recorded-input evaluation. Fresh GPU joint training completed two 17-slot episodes, and exact GPU recovery added one 17-slot episode. The final checkpoint completed another 17-slot evaluation.

Both evaluations preserved model and counter fingerprints. DQN, TD3 critics, actor and targets, and the timescale regulator performed training updates in the training/recovery runs. All episodes reached the configured horizon, had zero operating failures, and passed raw-component audits. The certified runs had zero accepted-deadline, node-temperature, and node-energy violations. Evaluation counters in the table are loaded training records; evaluation performs no learning updates.

## Tests and cross-platform numerical precision

The complete 59-test suite passed on the Windows runtime with no failure, error, or skip. The Linux-created example and Windows generator had a maximum received-signal-power relative difference of 7.20815e-15 across 17 slots. The fixture in `tests/test_public_link_outages.py` uses `rtol=1e-13` and `atol=0` for received signal power; bandwidth, TX/RX electrical powers, availability, and zero masks remain exact comparisons.

The test results and machine-readable verification are available in `evidence/windows_cuda_pytest.xml` and `evidence/windows_cuda_verification.json`.

## Checkpoint compatibility

The published models are the supplied CPU examples. Their evaluation checkpoint supports inference on compatible CPU/CUDA runtimes. Exact recovery requires matching source and runtime versions: the CPU recovery example uses its recorded CPU runtime, while these CUDA checks validate recovery of a freshly trained CUDA checkpoint. The GPU checkpoints used for this verification are separate from the published CPU model asset.
