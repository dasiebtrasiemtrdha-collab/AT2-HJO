# Parameter and architecture crosswalk

Reference: the revised manuscript and the released network/algorithm definitions. The table compares the normal training parameters with the quick functional profile.

| Parameter | Revised manuscript / formal profile | Functional model |
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

The formal actor and critic learning rates are 1e-4 and 3e-4, respectively. LT learning-rate sensitivity and the ST actor learning rate are distinct parameters.

## Formula references

| Formula | LaTeX label in the revised manuscript | Equation | Released implementation |
|---|---|---|---|
| General link SNR, for positive bandwidth | `eq:general_snr` | (2) | `communications/channels.py`: `snr`; source comment Eq(1) |
| UOWC channel gain | `eq:uowc_gain` | (4) | `communications/channels.py`: `optical_gain`; source comment Eq(3) |
| Link rate with availability and zero-bandwidth handling | `eq:general_rate` | (11) | `communications/channels.py`: `shannon_rate`; source comment Eq(10), stored `rate_model=shannon_equation_10` |
| Adaptive LT learning rate | `eq:adaptive_learning_rate` | (47) | `revision4/learning.py`: `Learner.dqn_update`; source comment Eq(45)-(46) |
| LT parameter update and periodic target synchronization | `eq:lt_parameter_update` | (48) | `revision4/learning.py`: `Learner.dqn_update`; source comment Eq(45)-(46) |

Source comments and stored configuration identifiers retain their release-time numbering. The labels above identify the corresponding formulas in the revised manuscript.

In `certified` mode, the ST reward contains the operating-cost terms without a proposal penalty; `rho` weights the actor auxiliary regularizer. In `resource_only` mode, the ST reward additionally subtracts `rho * proposal_phi`, and the same actor regularizer is retained. The quick example uses certified mode; each profile declares its execution mode.

Scheduling uses5*tanh(h); resource heads use tanh(h/2), then mask/project. head_temperature=2 is the resource divisor, not softmax temperature. Network summaries include DQN, actor, twin/target critics, attention policy and value critic with full state/action slices.

Current synthetic input statistics are provided in evidence/input_statistics.json and can be regenerated with scripts/validate_input_statistics.py.

Runtime: CPython3.12/PyTorch2.7.1 locks. Historical Python3.9 is not runtime-verified. At M30/B3/S6/K20, state dimension13773; full LT/ST state-pair replay arrays require approximately22.04GB together, excluding other storage and checkpoint overhead. The longer profiles are available for local training.
