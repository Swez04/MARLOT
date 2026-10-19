# MARLOT
## Multi-Agent Reinforcement Learning on Transactions

**MARLOT** is a modular, learning-focused project for adaptive fraud detection in synthetic payment transactions. It is being developed in phases, starting with a stateful transaction simulator and progressing toward anomaly detection, reinforcement learning, multi-agent coordination, and concept-drift experiments.

## Project Status

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Synthetic transaction simulation | Complete |
| **Phase 2** | Non-RL baseline detectors and evaluation | In progress |
| **Phase 3** | RL / MARL decision layer | Planned |
| **Phase 4** | Concept-drift experiments and scaling | Planned |

The current system generates synthetic transactions with ground-truth fraud labels, supports stateful fraud-event sequences, exposes a Python interface and REST API, and provides tools for testing and benchmarking the simulator.

---

## Features

### Phase 1 — Transaction Simulation

- Synthetic payment transactions with unique transaction IDs and reusable account, merchant, and device identities.
- Legitimate transactions and four initial fraud patterns:
  - **Amount spikes:** unusually large transactions relative to historical spending.
  - **Velocity attacks:** repeated transactions from the same account in a short period.
  - **Geo-hopping:** suspicious location changes within a short time.
  - **Merchant collusion:** multiple accounts repeatedly interacting with the same merchant.
- Persistent account and merchant state, including bounded transaction histories.
- Event-driven generation, where events control their own behavior, lifecycle, and timing.
- Configurable scenarios with event-selection weights.
- A shared simulation clock and reproducible random generation.
- Lazy transaction streaming, finite batch generation, and scenario-sequence simulation.
- A command-line interface for running simulations and exporting JSONL data.
- A FastAPI interface for requesting simulations over HTTP.
- Automated tests and repeatable performance benchmarks.

### Phase 2 — Baseline Fraud Detection

The initial baseline comparison includes:

- Rule-based detection.
- Isolation Forest for unsupervised anomaly detection.
- An autoencoder trained primarily on normal transaction behavior.
- Shared evaluation metrics for comparing detection performance.

The goal is to establish credible non-RL baselines before introducing learned decision policies.

---

## Setup

Use a stable Python release supported by the project's dependencies. The benchmark results documented below were collected using Python 3.12.0b4; rerun them on a stable release before treating them as a final performance baseline.

Create and activate a virtual environment.

**Windows — PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / macOS**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run commands from the repository root unless stated otherwise.

---

## Project Structure

```text
MARLOT/
├── simulator/
│   ├── __init__.py
│   ├── transaction.py
│   ├── state.py
│   ├── clock.py
│   ├── events.py
│   ├── scenarios.py
│   ├── generator.py
│   └── api/
│       ├── __init__.py
│       ├── app.py
│       └── schemas.py
│
├── detectors/
│   ├── rule_based.py
│   ├── isolation_forest.py
│   └── autoencoder.py
│
├── evaluation/
│   └── ...
│
├── scripts/
│   ├── generate_sample.py
│   └── benchmark_simulator.py
│
├── tests/
│   ├── test_transaction.py
│   ├── test_state.py
│   ├── test_events.py
│   ├── test_generator.py
│   ├── test_scenarios.py
│   ├── test_cli.py
│   └── test_api.py
│
├── notebooks/
├── benchmarks/
│   └── normal_100k.json
│
├── run_baseline.py
├── requirements.txt
├── pyproject.toml
├── README.md
└── .gitignore
```

The tree illustrates the intended organization; retain any additional files already present in your repository. Add modules as each phase requires them rather than creating empty folders for future components.

### Module responsibilities

| Module | Responsibility |
|---|---|
| `transaction.py` | Transaction data model and serialization. |
| `state.py` | Account and merchant histories and statistics. |
| `clock.py` | Centralized simulation-time management. |
| `events.py` | Legitimate and fraudulent event behavior. |
| `scenarios.py` | Scenario definitions and event-selection weights. |
| `generator.py` | Event orchestration, state updates, streaming, and batch generation. |
| `api/` | HTTP endpoints and request/response schemas. |
| `detectors/` | Non-RL fraud-detection baselines. |
| `evaluation/` | Metrics and comparisons across detectors. |
| `scripts/` | Developer utilities, CLI, and benchmarking. |
| `tests/` | Automated correctness and regression tests. |

---

## Generate Synthetic Transactions

### Command-line interface

Generate 1,000 transactions from the normal scenario:

```bash
python -m scripts.generate_sample \
    --scenario normal \
    --count 1000 \
    --seed 42
```

Generate a mixed-attack sample:

```bash
python -m scripts.generate_sample \
    --scenario mixed \
    --count 10000 \
    --seed 42
```

The scenario name must match a configuration in `simulator/scenarios.py`.

Supported arguments include:

| Argument | Description |
|---|---|
| `--scenario` | Select the simulation scenario. |
| `--count` | Number of transactions to generate. |
| `--seed` | Seed for reproducible simulation behavior. |
| `--start-time` | Explicit simulation start time in ISO-8601 format. |
| `--output` | Optional destination for a JSONL transaction export. |

The CLI prints a summary containing transaction counts, unique accounts and merchants, legitimate and fraudulent transactions, fraud-type breakdowns, and the timestamp range.

To export transactions:

```bash
python -m scripts.generate_sample \
    --scenario mixed \
    --count 1000 \
    --seed 42 \
    --output output/sample.jsonl
```

### Python interface

The generator can be reused directly in notebooks, evaluation scripts, and future RL environments:

```python
from datetime import datetime

from simulator.generator import TransactionGenerator

generator = TransactionGenerator(seed=42)

transactions = generator.generate_stream(
    n=1000,
    scenario_name="normal",
    start_time=datetime.fromisoformat("2026-01-01T00:00:00"),
)

print(f"Generated {len(transactions)} transactions")
```

Adjust keyword names to match the public signature in your current `generator.py`.

For lazy generation, use the streaming interface:

```python
from datetime import datetime

from simulator.generator import TransactionGenerator

generator = TransactionGenerator(seed=42)

transactions = generator.stream(
    scenario_name="normal",
    start_time=datetime.fromisoformat("2026-01-01T00:00:00"),
)

for transaction in transactions:
    # Process one transaction at a time.
    print(transaction.transaction_id)
    break

transactions.close()
```

The stream preserves its event, random-number-generator, and simulator-state context while the same iterator is consumed and resumed. The generator updates simulator state before yielding each transaction.

### Scenario sequences

The simulator also supports generating consecutive segments under different scenarios. This makes it possible to simulate changes in the transaction environment while preserving account history, merchant history, random state, and simulation time.

For example, a sequence might contain:

```text
Normal activity          1,000 transactions
Velocity-attack scenario   300 transactions
Mixed-attack scenario      700 transactions
                         ------------------
Total                    2,000 transactions
```

Use the scenario names and segment format defined by the existing `generate_scenario_sequence()` implementation. Segment counts refer to transactions, not events. The generator enforces exact segment lengths and switches scenarios at segment boundaries.

**Important:** Scenario weights are probabilities of selecting events when an event begins. They do not guarantee an identical percentage of fraudulent transactions in the final output.

---

## REST API

The simulator can also be accessed through FastAPI.

Start the server:

```bash
uvicorn simulator.api.app:app --reload
```

The API runs locally at `http://127.0.0.1:8000` by default.

### Available endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Check that the API is running. |
| `POST` | `/simulate` | Generate a finite simulation. |
| `GET` | `/docs` | Interactive Swagger documentation. |
| `GET` | `/openapi.json` | OpenAPI schema. |

Example health check:

```bash
curl http://127.0.0.1:8000/health
```

Example simulation request:

```bash
curl -X POST http://127.0.0.1:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "scenario": "normal",
    "count": 1000,
    "seed": 42,
    "start_time": "2026-01-01T00:00:00",
    "include_transactions": false
  }'
```

The simulation endpoint returns summary statistics by default. Set `include_transactions` to `true` when the response should include the individual transaction payloads.

Request validation rejects invalid scenarios, negative counts, malformed timestamps, and counts exceeding the configured request limit. Each request uses an independent generator, so simulator state does not accidentally leak between requests.

The API is designed for finite simulation requests at this stage. It does not yet provide a distributed or continuous network-streaming service.

---

## Phase 2 — Run the Baseline Comparison

Once the baseline detectors and evaluation modules are implemented, run:

```bash
python run_baseline.py
```

The intended workflow is:

1. Generate 2,000 normal transactions for fitting unsupervised detectors.
2. Generate a test sequence across configured scenarios covering amount spikes, velocity attacks, geo-hopping, and merchant collusion.
3. Run the rule-based detector, Isolation Forest, and autoencoder.
4. Compare their results using common evaluation metrics.

The initial evaluation reports:

- Precision.
- Recall.
- F1 score.
- False-positive rate.
- Throughput.

As the evaluation module develops, additional metrics should include PR-AUC and latency where appropriate.

### Evaluation principles

- Fit unsupervised detectors on normal training data where appropriate.
- Keep training and test data separate.
- Use transaction fraud labels as ground truth for evaluation, not as detector input.
- Compare detectors on the same test transactions.
- Report both detection quality and computational cost.
- Avoid interpreting a high accuracy score alone as evidence of good fraud detection, especially when fraud is rare.

---

## Run Tests

Run the full test suite from the repository root:

```bash
pytest
```

The current tests cover:

- Transaction construction and serialization.
- Account and merchant state updates.
- Bounded transaction and location histories.
- Event-specific behavioral invariants and event lifecycles.
- Exact transaction counts and event-boundary behavior.
- Timestamp ordering and centralized simulation time.
- Reproducibility under a fixed seed and explicit start time.
- Scenario configuration and validation.
- CLI argument handling and JSONL export.
- API validation, responses, and state isolation.

---

## Benchmark the Simulator

The simulator includes a benchmark script to measure lazy streaming and batch-generation throughput, with separate peak-memory measurements.

Example:

```bash
python -m scripts.benchmark_simulator \
    --scenario normal \
    --count 100000 \
    --seed 42 \
    --repeats 3 \
    --mode both \
    --output benchmarks/normal_100k.json
```

For a mixed workload:

```bash
python -m scripts.benchmark_simulator \
    --scenario mixed \
    --count 100000 \
    --seed 42 \
    --repeats 3 \
    --mode both \
    --output benchmarks/mixed_100k.json
```

The benchmark validates transaction counts, transaction-ID uniqueness, timestamp ordering, and merchant-state consistency. It reports median throughput and peak traced Python memory.

### Measured benchmark results

The following results were collected on Windows 11 using Python 3.12.0b4. Throughput values are medians across three repetitions.

| Scenario | Transactions | Streaming (txn/s) | Batch (txn/s) |
|---|---:|---:|---:|
| Normal | 1,000 | 82,139 | 102,446 |
| Normal | 10,000 | 62,889 | 90,474 |
| Normal | 100,000 | 96,220 | 101,545 |
| Mixed | 1,000 | 43,906 | 59,723 |
| Mixed | 10,000 | 70,026 | 25,567 |
| Mixed | 100,000 | 64,502 | 31,832 |

### Peak traced memory

| Scenario | Transactions | Streaming | Batch |
|---|---:|---:|---:|
| Normal | 1,000 | 1.42 MiB | 1.43 MiB |
| Normal | 10,000 | 6.84 MiB | 6.93 MiB |
| Normal | 100,000 | 41.61 MiB | 56.72 MiB |
| Mixed | 1,000 | 1.33 MiB | 1.33 MiB |
| Mixed | 10,000 | 6.58 MiB | 6.66 MiB |
| Mixed | 100,000 | 37.84 MiB | 52.72 MiB |

The tested workloads all exceeded the **10,000 transactions/sec simulator target**. The highest reported median throughput was 101,545 transactions/sec for normal batch generation at 100,000 transactions.

These numbers describe the synthetic simulation engine under the benchmark configuration. They do not represent end-to-end throughput for the REST API, fraud detectors, RL agents, or a production payment-processing system.

The Python environment used for these initial measurements was a beta release. Repeat benchmarks on a stable Python release before making final performance claims, and record the hardware and environment alongside the results. Peak memory reported by `tracemalloc` is Python-traced memory rather than total process RAM.

---

## Development Roadmap

### Phase 1 — Synthetic Transaction Simulation

**Status: Complete**

The simulator provides:

- A transaction data model.
- Stateful account and merchant histories.
- Event-driven legitimate and fraudulent activity.
- Configurable scenarios.
- Centralized simulation-time management.
- Lazy streaming and finite batch APIs.
- Scenario-sequence generation.
- CLI and REST interfaces.
- Automated tests and performance benchmarks.

### Phase 2 — Non-RL Baselines

**Status: In progress**

Develop and evaluate:

- Rule-based detection.
- Isolation Forest.
- Autoencoder-based anomaly detection.
- Shared evaluation metrics and baseline comparisons.

The objective is to establish reliable non-RL reference performance before adding a learned decision layer.

### Phase 3 — RL / MARL Decision Layer

**Status: Planned**

Introduce reinforcement learning to choose actions based on transaction features and detector risk signals.

The intended action space is:

- `ALLOW`
- `FLAG`
- `BLOCK`

Start with a single-agent RL baseline, then introduce regional agents and a global coordinator when the simpler environment is validated.

### Phase 4 — Concept Drift and Scaling

**Status: Planned**

Evaluate adaptation when fraud patterns change over time, including combinations of previously seen attacks. Investigate scaling and streaming infrastructure only after the simulator and detection baselines have been validated.

Kafka, Redis, Prometheus, Grafana, and distributed deployment are future infrastructure considerations, not requirements of the completed simulation milestone.

---

## Design Principles

MARLOT is being built incrementally, with an emphasis on modularity and testable behavior.

- **Separation of concerns:** the generator orchestrates events, events implement behavior, state records history, and scenarios configure event selection.
- **Reusable core:** simulation logic does not depend on FastAPI or infrastructure services.
- **Reproducibility:** seeded randomness and explicit simulation start times support controlled experiments.
- **Behavioral testing:** tests verify observable invariants rather than depending unnecessarily on internal implementation details.
- **Honest evaluation:** benchmarks and detection metrics are measured rather than assumed.
- **Incremental complexity:** new infrastructure and learning components are introduced only when the existing milestone is sufficiently tested.

The long-term objective is an adaptive fraud-detection system combining anomaly detection, reinforcement learning, multi-agent coordination, and an LLM supervisor. The current focus is building the simulator and baseline evaluation foundation needed to assess those components meaningfully.
