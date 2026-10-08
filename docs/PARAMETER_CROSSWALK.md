# Parameter and architecture crosswalk

Target: supplied TCOM_Revised_4_5.zip, main.tex SectionV-B and network/algorithm definitions. The table distinguishes normal training parameters and the quick functional profile.

| Parameter | Manuscript4.5 / formal profile | Functional model |
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

The final actorLR1e-4 agrees with the formal configuration. LT learning-rate sensitivity and the separate ST actor LR are distinct parameters.

Scheduling uses5*tanh(h); resource heads use tanh(h/2), then mask/project. head_temperature=2 is the resource divisor, not softmax temperature. Network summaries include DQN, actor, twin/target critics, attention policy and value critic with full state/action slices.

Current synthetic input statistics are provided in evidence/input_statistics.json and can be regenerated with scripts/validate_input_statistics.py.

Runtime: CPython3.12/PyTorch2.7.1 locks. Historical Python3.9 is not runtime-verified. At M30/B3/S6/K20, state dimension13773; full LT/ST state-pair replay arrays require approximately22.04GB together, excluding other storage and checkpoint overhead. The longer profiles are available for local training.
