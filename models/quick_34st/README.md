# Functional 34-ST checkpoints

Version4.0.2-public, seed2000, certified mode, two17-slot episodes. These are newly trained software examples, not the models that generated manuscript performance figures. Counters: environment34, critic27, actor13, target13, DQN4, TR5. Small batches LT2/ST4, warm-up LT2/ST8, policy delay2 differ from formal profiles.

From the project root:

```bash
python scripts/verify_files.py --root models/quick_34st --manifest models/quick_34st/FILE_MANIFEST.json
python -m at2hjo.cli evaluate --checkpoint models/quick_34st/evaluation.pt --seed 4000 --dataset data/example --device cpu --output runs/model_eval
python -m at2hjo.cli resume --checkpoint models/quick_34st/resume.pt --trust-checkpoint --episodes 3 --device cpu --output runs/model_resume
```

Recovery model origin: Linux CPython3.12.14, NumPy2.2.6, PyTorch2.7.1+cpu. Exact recovery requires matching source/configuration/runtime versions. For Windows CUDA use read-only evaluation or create a fresh local checkpoint. Evaluation uses restricted loading; recovery is explicitly trusted and must be checksum-verified. Raw episode records pass completion-time cost and log-completeness checks.

Both distribution ZIPs have the top-level directory AT2HJO. Extract the optional model asset beside the source distribution; its files belong under models/quick_34st. The source archive contains these metadata/logs but the large .pt files are available only from the separate model asset.
