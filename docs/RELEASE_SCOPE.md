# Core implementation release

This core release provides the AT2-HJO implementation accompanying revised manuscript 4.5: the native simulator, improved DQN, improved TD3, attention-based timescale regulator, network definitions, training/evaluation/recovery scripts, parameter configurations, random seeds, synthetic input generators, and executable examples.

The core includes general training and evaluation interfaces and M10/M20/M30 profiles. Experiment-specific runners for timing/scaling, unseen-scenario adaptation, ablation studies, training-interaction analysis, and external comparison methods are not included in this release. `EXPERIMENT_MAP.csv` identifies the available interfaces and the scope of each experiment family.

Public release of additional project materials requires institutional approval. Following acceptance, we will initiate the release-review procedure and publish the approved experimental materials in this same repository as a versioned release.

## Executable example

Software version: **4.0.2-public**. State/simulator identity: `tcom_4_0_fifo_certificate_v1`.

The quick example contains two 17-slot training episodes: 34 environment steps, 27 critic updates, 13 actor updates, 13 target updates, 4 DQN updates, and 5 regulator updates. Data, configuration, network summaries, and logs are included. Evaluation and recovery checkpoints are supplied in the separate model asset. The supplied profiles also support training with the normal parameters.

## Verification

The 59-test suite passes under the locked Linux CPython 3.12 CPU runtime and the verified Windows CUDA runtime. Short CPU/CUDA training, evaluation, and episode-boundary recovery pass the raw-log and component checks. The CUDA runs include 34 training slots, a 17-slot recovery extension, and 17-slot evaluations; details and runtime compatibility requirements are in `docs/GPU_VERIFICATION.md`.

Validation seeds are accepted and their roles are recorded. BN blockage and RF outage settings are active, and the default input realizations are preserved. Device diagnostics report the CUDA version, device count, and names; learning modules share CPU/CUDA selection. Evaluation uses restricted loading, and recovery requires explicit trust and matching source/runtime. Logs are verified before checkpoint creation.

This verification covers short CPU/CUDA training, evaluation, and recovery runs. Full-scale training and multi-seed performance evaluation are outside its scope.

The root `README.md` describes the published directories and executable commands. Approved additional materials will be versioned in this repository, so the manuscript availability link can continue to use the same repository address.
