# MARLOT

Adaptive fraud detection, built up in phases. Currently at:

**Phase 1** — synthetic transaction simulator (`simulator/`)
**Phase 2** — non-RL baseline detectors (`detectors/`) + evaluation (`evaluation/`)

Not yet built (add folders as you get here):
- **Phase 3** — RL / MARL decision layer (`agents/`, `rl/`)
- **Phase 4** — concept-drift regime experiments, scaled up

## Setup

```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the baseline comparison

```bash
python run_baseline.py
```

This will:
1. Generate 2000 "normal" transactions to fit the unsupervised detectors on.
2. Generate a mixed test stream that cycles through several fraud regimes
   (amount spikes, velocity attacks, geo-hopping, merchant collusion).
3. Run the rule-based, Isolation Forest, and autoencoder detectors on the
   test stream.
4. Print precision / recall / F1 / false-positive rate / throughput for each.

## Folder guide

```
simulator/    transaction data model + synthetic generator
detectors/    rule-based, Isolation Forest, autoencoder baselines
evaluation/   shared metrics (precision/recall/F1/PR-AUC/latency/throughput)
notebooks/    scratch space for exploring data & plotting results
run_baseline.py   single script that runs everything above end-to-end
```

## Next steps (once baselines work)

- Add a `regime` comparison table like the one in the design doc: run each
  detector across regimes 1–5 and see how much recall degrades under drift.
- Only then start `agents/` for the RL response agent (single-agent DQN
  first, before going multi-agent).
- Hold off on Kafka/Redis/Kubernetes until the RL loop works on data
  already in memory — infra can wait.
