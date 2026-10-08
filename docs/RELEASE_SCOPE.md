# Core implementation release

This repository provides the AT2-HJO core implementation accompanying revised manuscript 4.5: the native simulator, improved DQN, improved TD3, attention-based timescale regulator, network definitions, training/evaluation/recovery scripts, parameter configurations, random seeds, synthetic input generators and executable examples.

Public release of additional project materials requires institutional approval. Following acceptance, we will initiate the release-review procedure and publish the approved experimental materials in this same repository as a versioned release.

## Current executable example

Software version: 4.0.2-public. State/simulator identity: tcom_4_0_fifo_certificate_v1. The quick example contains two17-slot training episodes, with34environment steps,27critic updates,13actor and target updates,4DQN updates and5regulator updates. Data, configuration, network summaries, logs and evaluation/recovery checkpoints are provided. Normal-parameter training can be run using the supplied profiles.

## Engineering checks

- 59 tests pass under the lockedLinuxCPython3.12CPU runtime.
- Example training, read-only evaluation and episode-boundary recovery pass the raw-log and component checks.
- Validation seeds are accepted and their roles are recorded.
- BN blockage and RF outage settings are active; the default input realizations are preserved.
- CUDA diagnostics include version, device count and names; CPU/CUDA selection is shared by learning modules.
- Evaluation uses restricted loading; recovery requires explicit trust and matching source/runtime.
- Logs are verified before successful completion and checkpoint creation.

The current published directories are listed in the rootREADME. Future approved materials will be versioned in this repository; the manuscript availability link can continue pointing to the same repository.
