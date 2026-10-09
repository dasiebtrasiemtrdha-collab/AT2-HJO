# Implementation reference

Default training parameters and quick-example settings.

| Parameter | Formal profile | Quick example |
|---|---|---|
| Actor LR | 1e-4 | same |
| Critic/regulator LR | 3e-4 / 3e-4 | same |
| Adaptive LT LR | [1e-5,3e-4] | same |
| LT/ST batch | 128/128 | 2/4 |
| Episodes/ST slots | 2500/500 | 2/17 |
| ST/LT warmup | 1000/128 transitions | 8/2 |
| Effective critic-update policy delay | 16 | 2 |
| Replay/gamma | 100000/.99 | same |
| Tau/rho/weights | .005/5/(.5,.3,.2) | same |
| Prior/epsilon_eta | (.3,.4,.3)/.01 | same |
| MLP/attention history/width | 256-256/4/64 | same |
| Mode | Figures: resource_only; checks: certified | certified |
| M/B/S/K | 10-30/3/6/20 | 30/3/6/20 |

The adaptive LT learning-rate range and the ST actor learning rate are separate settings.

## Formulas and functions

| Formula | LaTeX label | Equation | Implementation |
|---|---|---|---|
| General link SNR, for positive bandwidth | `eq:general_snr` | (2) | `communications/channels.py`: `snr` |
| UOWC channel gain | `eq:uowc_gain` | (4) | `communications/channels.py`: `optical_gain` |
| Link rate with availability and zero-bandwidth handling | `eq:general_rate` | (11) | `communications/channels.py`: `shannon_rate` |
| Adaptive LT learning rate | `eq:adaptive_learning_rate` | (47) | `revision4/learning.py`: `Learner.dqn_update` |
| LT parameter update and periodic target synchronization | `eq:lt_parameter_update` | (48) | `revision4/learning.py`: `Learner.dqn_update` |

The actor loss in both modes is `-mean(Q) + rho * proposal_phi`. The ST reward uses the operating-cost terms in `certified`; `resource_only` additionally subtracts `rho * proposal_phi`.

Scheduling uses5*tanh(h); resource heads use tanh(h/2), then mask/project. head_temperature=2 is the resource divisor, not softmax temperature. Network summaries include DQN, actor, twin/target critics, attention policy and value critic with full state/action slices.

Current synthetic input statistics are provided in evidence/input_statistics.json and can be regenerated with scripts/validate_input_statistics.py.

Runtime: CPython 3.12 with the PyTorch 2.7.1 dependency locks. At M30/B3/S6/K20, state dimension13773; full LT/ST state-pair replay arrays require approximately22.04GB together, excluding other storage and checkpoint overhead. The longer profiles are available for local training.
