# Windows / CUDA verification for AT2-HJO 4.0.2

The current4.0.2 source archive and34-ST CPU example were checked on Windows
using the supplied wheel installed outside the research repository. All187
source-distribution manifest entries and15model entries were checked before
testing. Native Python source and line endings, normal configuration, example
data and the supplied CPU checkpoint binaries remain unchanged.

Runtime: Python3.12.14, NumPy2.2.6, PyTorch2.7.1+cu126, CUDA12.6,
SciPy1.15.3, one NVIDIA GeForceRTX4060. Selected device: cuda:0.
Source hash: `397d12fc0031f6202aa7547c5b67bd2b5e5a678c0c06915df8c95426cb9aaf50`.

## Actual short GPU execution

| Run | Seed | Slots this invocation | Cumulative training slots | critic/actor/target | DQN/TR | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| provided_cpu_model_eval | 4000 | 17 | 34 | 27/13/13 | 4/5 | 2.24 |
| train | 2000 | 34 | 34 | 27/13/13 | 4/5 | 25.64 |
| resume | 2000 | 17 | 51 | 44/22/22 | 6/7 | 15.01 |
| eval | 4000 | 17 | 51 | 44/22/22 | 6/7 | 1.64 |

The supplied CPU evaluation checkpoint loaded57tensors onto CUDA and completed
one17-slot recorded-input evaluation. Fresh GPU joint training completed two
17-slot episodes, then exact GPU recovery added one17-slot episode. Its final
checkpoint completed another17-slot read-only evaluation. Both evaluations
preserved model/counter fingerprints; DQN, TD3critics/actor/targets and TR
performed actual training updates. All episodes reached the configured
horizon, had zero operating failures and passed raw-component audits. All
certified runs had zero accepted-deadline, node-temperature and node-energy
violations. Evaluation counters above are loaded training records, not new
evaluation updates.

## Tests and the portability correction

The initial Windows test run had58passes and one fixture-comparison failure.
The Linux-created example and Windows generator differed only in the last
bits of received signal power: maximum relative difference7.20815e-15 across
17slots; bandwidth, TX/RX electrical powers, availability and zero masks were
exactly equal. Only tests/test_public_link_outages.py changed: signal power
uses rtol1e-13/atol0, while all other fields and zero masks stay exact. The
original file's line endings were preserved. No physical equation, execution
certificate tolerance, parameter, trace bytes or model was changed.

After this test-only correction, the complete59-test suite passed with no
failure, error or skip. The original failure receipt remains in local
diagnostics. A sanitized finalXML and machine-readable verification are in
evidence/windows_cuda_pytest.xml and evidence/windows_cuda_verification.json.

## Scope

The supplied CPU examples remain the published models; the new GPU training
checkpoints only support this local verification. CPU evaluation models can
be used for GPU inference. Exact resume retains its matching source/runtime
requirement; the CPU-only resume example should use its recorded CPU runtime,
and this report validates recovery of a freshly createdCUDA checkpoint.

No formal2500-episode,20-seed or additional5000-ST run was performed. These
are short implementation checks, not manuscript performance reproduction.
No paper, response, footnote, institutional statement or project license was
changed. This report does not perform GitHub publication.
