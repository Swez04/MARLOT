## Project: Adaptive MARL System for Real-Time Fraud Detection

### 1. Goal

Build a **learning-focused multi-agent reinforcement learning system** that simulates real-time payment fraud detection and learns to:

* Process **10K+ transactions/sec** in simulation.
* Detect both known and evolving fraud patterns.
* Minimize fraud losses while controlling false positives/customer friction.
* Demonstrate adaptation to **concept drift** without manually retraining the entire system.
* Learn how **anomaly detection + RL + MARL + LLM supervision + streaming infrastructure** fit together.

This is a **learning/engineering project**, not intended primarily as a research publication.

---

# 2. Overall Architecture

```text
Synthetic Transaction Generator
            ↓
      Kafka / Redis Stream
            ↓
    Feature / State Builder
            ↓
 ┌──────────┼──────────┐
 ↓          ↓          ↓
India       USA      Europe
Detector   Detector   Detector
 ↓          ↓          ↓
India RL   USA RL    Europe RL
Agent      Agent     Agent
 └──────────┼──────────┘
            ↓
    Global Coordinator
         RL Agent
            ↓
     System Metrics
            ↓
     LLM Supervisor
            ↓
   Recommendations /
      Hypotheses
            ↓
    Simulation Testing
            ↓
  Accept / Reject Change
            ↓
    Updated RL System
```

---

# 3. Transaction Simulator

Create a synthetic payment ecosystem containing:

* Accounts
* Merchants
* Devices
* Regions
* Merchant categories
* Locations
* Transaction histories
* Timestamps
* Amounts

Each transaction should contain approximately:

```text
transaction_id
account_id
merchant_id
region
amount
timestamp
location
merchant_category
device_id
fraud_label
fraud_type
```

The simulator provides **ground-truth fraud labels** so the RL environment can calculate rewards.

### Fraud patterns

Implement at least:

1. **Amount spikes**

   * Normally small transactions followed by unusually large payments.

2. **Velocity attacks**

   * Many transactions from an account in a short period.

3. **Geo-hopping**

   * Impossible/unusual geographic movement.

4. **Merchant/account collusion**

   * Many accounts showing coordinated behavior around specific merchants.

Later introduce new combinations/patterns to create **concept drift**.

---

# 4. Streaming Layer

Initially, transactions can be passed directly into the simulator.

Then add:

* **Kafka** for transaction streaming.
* **Redis** for fast rolling state.

Target:

> **10K+ transactions/sec simulated throughput**

Measure:

* TPS
* p50 latency
* p95 latency
* p99 latency

---

# 5. Feature / State Builder

Maintain transaction-level and historical features.

Examples:

```text
amount
transaction hour
merchant category
geo risk
merchant risk
velocity over 1 min
velocity over 10 min
average transaction amount
amount deviation
new device
account history
merchant/account concentration
```

This state feeds the anomaly detectors and RL agents.

---

# 6. Anomaly Detection Layer

Use two independent models:

### Isolation Forest

Produces an anomaly score:

```text
IF score = 0.87
```

### Autoencoder

Train primarily on normal behavior and use reconstruction error as anomaly score:

```text
AE score = 0.91
```

Combine them with additional rules/features into a **risk representation**, e.g.:

$$
Risk=w_1IF+w_2AE+w_3Rules
$$

Important:

> The risk score is **not the final fraud decision**. It is information given to the RL agent.

---

# 7. Regional RL Agents

Create one RL decision-making agent per region:

```text
India RL Agent
USA RL Agent
Europe RL Agent
...
```

Each agent receives transactions belonging to its region.

### Observation/state

Contains:

```text
transaction features
rolling statistics
IF score
autoencoder score
risk score
geo/merchant risk
historical context
```

### Action space

```text
ALLOW
FLAG
BLOCK
```

### RL loop

```text
State
  ↓
RL Agent
  ↓
Action
  ↓
Simulator/environment
  ↓
True outcome
  ↓
Reward
  ↓
Policy update
```

Mathematically:

$$
s_t\rightarrow a_t\rightarrow r_t,s_{t+1}
$$

Start with **DQN** for the regional agents before attempting more sophisticated MARL algorithms.

---

# 8. Reward Function

The reward should represent **business value**, not merely classification accuracy.

Basic structure:

$$
R =
V_{TP}
-C_{FP}
-C_{FN}
-C_{friction}
+B_{early}
$$

Example:

```text
Block + fraud      → large positive reward
Block + legitimate → large negative reward
Allow + fraud      → large negative reward
Allow + legitimate → small positive reward
Flag + fraud       → moderate positive reward
Flag + legitimate  → small negative reward
```

The exact values should be configurable so you can experiment with the fraud-catching vs false-positive tradeoff.

---

# 9. Global Coordinator RL Agent

The coordinator **does not decide individual transactions**.

It observes the overall behavior of the regional agents.

Its state may include:

```text
regional fraud recall
regional precision
regional false-positive rate
regional rewards
block/flag/allow rates
fraud losses
customer friction
throughput
emerging fraud statistics
```

It optimizes the **global objective**.

Conceptually:

```text
India RL ──┐
USA RL ────┼──→ Global Coordinator
Europe RL ─┘
```

The coordinator can influence higher-level policy parameters such as:

* regional risk sensitivity
* reward weighting
* exploration/exploitation behavior
* coordination parameters
* threshold/weight configurations

The exact coordinator action space should be kept small initially.

---

# 10. MARL Objective

The regional agents optimize local decisions while the coordinator optimizes global performance.

Example global objective:

$$
R_{global}
=
\sum_i R_i
-\lambda FP_{global}
$$

The system should balance:

```text
Fraud caught
        +
Fraud loss prevented
        -
False positives
        -
Customer friction
```

---

# 11. System Monitoring

Continuously aggregate:

```text
transaction volume
fraud rate
fraud recall
precision
false-positive rate
fraud loss
customer friction
regional performance
agent rewards
block/flag/allow rates
risk distributions
top suspicious merchants/accounts
fraud-type distribution
latency
throughput
```

Use:

* **Prometheus**
* **Grafana**

for monitoring and visualization.

---

# 12. Concept Drift

The simulator must deliberately change fraud behavior over time.

Example:

```text
Phase 1 → amount spikes
Phase 2 → velocity attacks
Phase 3 → geo-hopping
Phase 4 → merchant collusion
Phase 5 → new combinations / unseen patterns
```

Evaluate whether the RL agents can adapt rather than simply memorizing the original fraud distribution.

Track performance before, during, and after each regime change.

---

# 13. LLM Supervisor

The LLM operates **outside the real-time transaction path**.

It does **not** examine individual transactions and does **not** directly block transactions.

Instead, periodically—e.g. every 30 minutes—it receives an aggregated system summary:

```text
Global metrics
Regional metrics
Fraud trends
False-positive trends
Agent rewards
Risk distributions
Suspicious merchants/accounts
Emerging patterns
Recent policy changes
```

Its role is:

> **Analyze the system and suggest hypotheses or improvements.**

Example:

```text
Europe:
FPR ↑ 40%
Recall unchanged

Merchant M184:
Velocity ↑ 340%
Unique accounts ↑ 280%

LLM hypothesis:
Possible coordinated merchant fraud.
Investigate network-level features.
```

---

# 14. LLM → Experiment → RL

The LLM should **not directly modify production policies**.

Instead:

```text
LLM
 ↓
Recommendation
 ↓
Candidate policy/reward change
 ↓
Run in simulator
 ↓
Compare against current policy
 ↓
Accept / reject
 ↓
Update RL system if beneficial
```

Example:

```text
LLM:
Increase merchant-network feature importance.

Simulator:

Current:
Recall = 91%
FPR = 3.4%
Reward = 0.73

Candidate:
Recall = 94%
FPR = 3.7%
Reward = 0.81

→ Candidate accepted
```

This is the main **agentic component** of the project.

---

# 15. Technology Stack

### Core

* Python
* PyTorch
* NumPy
* Pandas
* scikit-learn

### RL

* **Ray RLlib** preferred for the eventual MARL implementation
* Gymnasium environment
* DQN initially
* Later explore QMIX / centralized-critic approaches

### Streaming

* Kafka
* Redis

### LLM orchestration

* LangGraph or a lightweight custom supervisor

### Monitoring

* Prometheus
* Grafana

### Deployment

* Docker
* Kubernetes/minikube **only after the core system works**

---

# 16. Development Milestones

### Milestone 1 — Fraud simulator

```text
Synthetic accounts/merchants
        ↓
Normal transactions
        +
4 fraud types
```

**Deliverable:** controllable transaction generator with ground truth.

---

### Milestone 2 — Detection

```text
Transactions
 ↓
Isolation Forest
 ↓
Autoencoder
 ↓
Risk representation
```

**Deliverable:** working anomaly-detection baseline.

---

### Milestone 3 — Single-agent RL

```text
Risk representation
       ↓
DQN
       ↓
Allow / Flag / Block
       ↓
Reward
       ↓
Learning
```

**Deliverable:** understand and demonstrate the complete RL loop.

---

### Milestone 4 — Regional MARL

```text
India Agent
USA Agent
Europe Agent
```

**Deliverable:** multiple cooperating RL agents with regional specialization.

---

### Milestone 5 — Global coordinator

```text
Regional agents
      ↓
Coordinator RL
      ↓
Global optimization
```

**Deliverable:** hierarchical/cooperative MARL system.

---

### Milestone 6 — Concept drift

Introduce changing fraud regimes.

**Deliverable:** demonstrate adaptation to evolving attacks.

---

### Milestone 7 — LLM supervisor

```text
System metrics
      ↓
LLM
      ↓
Hypothesis/recommendation
```

**Deliverable:** LLM capable of identifying system-level issues and proposing changes.

---

### Milestone 8 — Closed-loop supervision

```text
LLM recommendation
       ↓
Simulation experiment
       ↓
Evaluation
       ↓
Accept/reject
       ↓
RL policy/configuration update
```

**Deliverable:** genuinely agentic feedback loop.

---

### Milestone 9 — Real-time infrastructure

Add:

```text
Kafka
Redis
Prometheus
Grafana
```

and benchmark:

```text
1K TPS
5K TPS
10K TPS
25K TPS+
```

**Deliverable:** demonstrate the real-time architecture and 10K+ TPS target.

---

# 17. Final Project

The finished system should demonstrate this complete loop:

```text
              SYNTHETIC PAYMENT WORLD
                       ↓
                  Transactions
                       ↓
                 Kafka / Redis
                       ↓
                Feature Builder
                       ↓
             IF + Autoencoder
                       ↓
                 Risk Signals
                       ↓
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
    India RL        USA RL        Europe RL
        └──────────────┼──────────────┘
                       ↓
              Global RL Coordinator
                       ↓
                Fraud Decision
              /       |        \
           ALLOW     FLAG     BLOCK
                       ↓
                  Environment
                       ↓
                    Reward
                       ↓
                 RL Learning
                       ↓
                 System Metrics
                       ↓
                LLM Supervisor
                       ↓
             Hypothesis / Proposal
                       ↓
                  Simulation
                       ↓
             Validate improvement
                       ↓
              Update RL system
                       ↓
                Continue learning
```

### The core idea to remember

**IF + Autoencoder:** *"How suspicious is this?"*

**Regional RL agents:** *"What should I do about it?"*

**Global RL coordinator:** *"How should all the regional agents work together?"*

**LLM supervisor:** *"What is happening to the overall system, and what should we investigate or improve?"*

**Simulator:** *"What actually happened, and therefore what reward should the RL system receive?"*

That is the complete context needed to start developing the project.
