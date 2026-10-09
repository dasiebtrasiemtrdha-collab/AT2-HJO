# Core implementation release

The package contains the AT2-HJO simulator, improved DQN, improved TD3, attention-based timescale regulator, network definitions, training/evaluation/recovery scripts, parameter configurations, random seeds, synthetic input generators and executable examples.

Training and evaluation interfaces include M10/M20/M30 profiles. `EXPERIMENT_MAP.csv` lists the entry points for the experiments and the dedicated runners they require.

## Executable example

Software version: **4.0.2-public**. State/simulator identity: `tcom_4_0_fifo_certificate_v1`.

The quick example contains two 17-slot training episodes: 34 environment steps, 27 critic updates, 13 actor updates, 13 target updates, 4 DQN updates, and 5 regulator updates. Data, configuration, network summaries, and logs are included. Evaluation and recovery checkpoints are supplied in the separate model asset. The supplied profiles also support training with the normal parameters.

## Verification

The 59-test suite passes under the locked Linux CPython 3.12 CPU runtime and the verified Windows CUDA runtime. Short CPU/CUDA training, evaluation, and episode-boundary recovery pass the raw-log and component checks. The CUDA runs include 34 training slots, a 17-slot recovery extension, and 17-slot evaluations; details and runtime compatibility requirements are in `docs/GPU_VERIFICATION.md`.

Runs record training, validation and test seed roles. The input generator supports BN blockage and RF outage settings. Device diagnostics report the CUDA version, device count and names. Evaluation uses restricted loading; recovery requires explicit trust and matching source/runtime versions.

See the root `README.md` for installation, executable commands and materials availability.
