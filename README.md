# AT2-HJO core implementation

Core implementation accompanying **Thermal-Aware Adaptive Two-Timescale Service Deployment and Task Scheduling in Buoy-Satellite Cooperative Marine Edge Computing Networks** (TCOM-TPS-26-1840, revised manuscript).

Software **4.0.2-public**; simulator/state identity `tcom_4_0_fifo_certificate_v1`.

This repository provides the AT2-HJO core implementation, including the simulator, improved DQN, improved TD3, attention regulator, training/evaluation/recovery scripts, complete network definitions, configurations, random seeds, synthetic input generators, example datasets and executable checkpoints. The current quick example contains **two 17-slot training episodes (34 ST steps)**.

Following acceptance, we will initiate the institutional release-review procedure and publish the approved experimental materials in this repository as a versioned release.

## Repository contents

| Path | Contents |
|---|---|
| `src/at2hjo/revision4/` | Simulator, physical/deployment/queue execution, networks, learners, data and reporting |
| `src/at2hjo/geometry/`, `communications/`, `device.py` | Geometry, channels and CPU/CUDA selection |
| `configs/revision4/` | Functional, development and full-scale training profiles, physical parameters and seed roles |
| `data/example/`, `data/example_open/` | Hash-verified small regular and stochastic/open synthetic inputs |
| `tests/`, `scripts/` | Tests, installation, input-statistics checks and file verification |
| `models/quick_34st/` | Functional-run logs and metadata; optional `.pt` files from the model asset |
| `evidence/`, `docs/` | Verification receipts, scope, parameter and architecture documentation |
| `PARAMETERS.csv`, `EXPERIMENT_MAP.csv`, `BASELINE_SCOPE.json` | Parameters, experiment entry points and comparison-material scope |
| `requirements/`, `third_party/` | Dependency locks, notices and dependency licenses |

The package includes the 34-ST example, 59-test results and short CPU/CUDA run records. Full-scale training profiles and commands are provided. Timing, adaptation, ablation, convergence and external comparison experiments require separate runners; their full-scale result artifacts are outside this core package. See `EXPERIMENT_MAP.csv` for the experiment entry points.


## Installation

Use **CPython 3.12**. The standalone package is verified with Python 3.12 and the dependency locks below.

Linux CPU:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/cpu-linux-py312.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python -m at2hjo.cli doctor --device auto
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements/windows-cpu-py312.lock
.venv/Scripts/python -m pip install --no-deps -e .
.venv/Scripts/python -m at2hjo.cli doctor --device auto
```

For NVIDIA CUDA, use `requirements/windows-cuda126-py312.lock` in Windows. Use the installed environment's Python for the commands below. Device options are `auto`, `cpu`, `cuda`, `cuda:0`; auto selects available CUDA or CPU. Doctor reports CUDA availability, PyTorch CUDA version, device count and names. Physical arrays and replay storage stay in host RAM; networks/batches use the selected device. Dependency installation requires network access.

## Functional training, evaluation and recovery

Use a new output directory for each invocation:

```bash
python -m at2hjo.cli train --config configs/revision4/smoke.json --seed 2000 --device cpu --output runs/quick_joint
python -m at2hjo.cli evaluate --checkpoint runs/quick_joint/evaluation.pt --seed 4000 --dataset data/example --device cpu --output runs/quick_eval
python -m at2hjo.cli audit --run runs/quick_joint --output runs/quick_joint/audit.json
python -m at2hjo.cli resume --checkpoint runs/quick_joint/resume.pt --trust-checkpoint --episodes 3 --device cpu --output runs/continued_joint
python -m pytest tests -q -p no:cacheprovider
```

Resume `--episodes` is a cumulative target. Evaluation uses restricted tensor/primitive loading. Recovery contains additional NumPy/Python RNG/replay state and requires `--trust-checkpoint` for a trusted, checksum-verified checkpoint. Exact recovery requires matching source and Python/NumPy/PyTorch version strings; CPU and CUDA builds differ. Evaluation or a fresh local training run can be used when recovery versions differ.

`--algorithm improved-dqn`, `improved-td3`, `at2-hjo` selects LT, ST, joint operation. The quick profiles use smaller batches and warm-up budgets; settings are listed in `docs/PARAMETER_CROSSWALK.md`.

## Optional functional models

Download [**AT2HJO_QuickModels_34ST_4.0.2.zip**](https://github.com/dasiebtrasiemtrdha-collab/AT2-HJO/releases/download/v4.0.2/AT2HJO_QuickModels_34ST_4.0.2.zip). This 34-ST example supports installation, inference and recovery checks. Extract the source and model distributions in the same parent directory; both have the top-level folder `AT2HJO`. Checkpoints then appear at `models/quick_34st/`.

```bash
python scripts/verify_files.py --root models/quick_34st --manifest models/quick_34st/FILE_MANIFEST.json
python -m at2hjo.cli evaluate --checkpoint models/quick_34st/evaluation.pt --seed 4000 --dataset data/example --device cpu --output runs/supplied_model_eval
python -m at2hjo.cli resume --checkpoint models/quick_34st/resume.pt --trust-checkpoint --episodes 3 --device cpu --output runs/supplied_model_resume
```

The supplied recovery model uses the Linux CPU dependency lock. Exact recovery requires its matching runtime; for inference on compatible CPU/CUDA runtimes, use the evaluation model. Verify source with `python scripts/verify_files.py --root . --manifest FILE_MANIFEST.json`.

## Synthetic inputs and seeds

```bash
python -m at2hjo.cli validate-data --dataset data/example
python -m at2hjo.cli validate-data --dataset data/example_open
python -m at2hjo.cli generate --config configs/revision4/stochastic_smoke.json --seed 4000 --output data/generated4000
python scripts/validate_input_statistics.py --output results/input_statistics.json
```

`trace.npz` contains `[slot, source, destination, field]` links (received signal W, bandwidth Hz, TX electrical W, RX electrical W, availability) and exogenous drivers. `requests.jsonl` stores offered requests: input bits, CPU cycles, service ID, deadline s, arrival slot; IDs/slots are zero-based. Manifests provide shapes, units, seeds and hashes. Materialized examples drive one episode-0 evaluation.

Named streams use SHA-256 of `seed/episode/stream_name`; learner/replay streams are separate and saved for recovery. Formal roots are training1000-1002, validation2000-2004, paired-test0-19. Development roles are independently labelled. Evaluation accepts formal validation/test roots and records their role. Current generator checks appear in `evidence/input_statistics.json`.

## Architecture and full-scale training profiles

MLPs use 256-256 ReLU. TD3 has scheduling/CPU/UOWC/RF heads, twin critics and targets. DQN scores state/candidate pairs with a thermal-aware prior. The regulator has single-head attention (history4, width64), interval policy and value critic. Complete native definitions are in `networks.py`; each run writes `network_summary.json` and `state_layout.json`. Other network sizes require independently initialized compatible networks.

Formal defaults: actorLR1e-4, critic/regulatorLR3e-4, DQN range[1e-5,3e-4], batch128, replay100000, gamma.99, tau.005, rho5, weights(.5,.3,.2). Execution includes persistent queues/transfers, executed-action critics, continuous-only target smoothing, actual-duration SMDP discount and terminal masks. Resource logits use `tanh(h/2)`; `head_temperature=2` is the divisor, not a softmax temperature.

`resource_only` is the regular-load projected-action soft-constraint benchmark. `certified` adds joint hard execution checks. The quick example uses certified mode; profile JSON files explicitly identify their execution mode.

```bash
python -m at2hjo.cli train --config configs/revision4/formal_resource_only.json --seed 1000 --device auto --output runs/formal_soft1000
python -m at2hjo.cli train --config configs/revision4/formal_certified.json --seed 1000 --device auto --output runs/formal_certified1000
```

The full-scale training profiles use 2500 episodes and 500 slots. At M30/B3/S6/K20, state dimension13773 and full-capacity LT/ST state-pair arrays alone require about22.04GB together. Candidates/actions/Python objects, models, optimizers and atomic checkpoint copies add overhead: plan approximately48-64GB host RAM and sufficient disk. CUDA does not remove this host-memory requirement.

Runs write raw slots/intervals, metrics, counters, source/config identities and checkpoints. `aggregate` and `plot` process those run outputs.

## Scope, citation and license

See `docs/RELEASE_SCOPE.md`, `docs/PARAMETER_CROSSWALK.md` and `CITATION.md`. Cite the repository and corresponding version when using this implementation. Dependency notices are provided in `third_party/`.
