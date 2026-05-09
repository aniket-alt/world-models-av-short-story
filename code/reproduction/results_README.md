# Code Reproduction — Phase 3
## Latent Trajectory Forecasting via Autoresearch Loop

**Paper:** Latent World Models for Automated Driving: A Unified Taxonomy, Evaluation Framework, and Open Challenges  
**ArXiv:** [2603.09086](https://arxiv.org/abs/2603.09086)  
**Course:** CMPE 258 — Deep Learning, SJSU

---

## What This Reproduces

The paper's central claim in Section II-B (Latent-Centric Planning & RL) is that **latent representations compress high-dimensional observations and enable temporally coherent future state prediction**. Models like Think2Drive, LatentDriver, and LAW all encode past driving observations into a compact latent vector and predict future states from that vector.

This experiment directly demonstrates that concept by running an **autoresearch loop** — inspired by the `dlmastery/autoresearch` template — across three latent encoder architectures on a vehicle trajectory forecasting task.

---

## The Autoresearch Loop

Following the autoresearch philosophy (one metric, one search space, keep/discard):

```
METRIC    : ADE — Average Displacement Error (meters, lower = better)
            Same as L2 Displacement Error used in the paper (Section IV)

SEARCH    : Three latent encoder architectures
SPACE       1. LSTM  — Continuous hidden state (Think2Drive-style)
            2. GRU   — Gated continuous state (DreamerV3-style)
            3. Transformer — Attention context vector (LAW/GenAD-style)

LOOP      : For each architecture:
              → Train encoder on past trajectory (8 timesteps = 4s)
              → Decode latent → predict future positions (12 timesteps = 6s)
              → Evaluate ADE on held-out test set
              → KEEP if ADE improves over best so far
              → DISCARD otherwise
              → Log result
```

---

## Setup & Running

**Step 1 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2 — Run the experiment**
```bash
python train.py
```

Expected runtime: ~3–5 minutes on CPU, ~1 minute on GPU.

No dataset download needed — the script generates synthetic trajectory data matching real driving statistics (nuScenes-style: 2Hz sampling, urban speeds 3–12 m/s).

---

## Results

See `results/metrics.json` for full results and autoresearch log.

| Architecture | ADE ↓ | FDE ↓ | Params |
|---|---|---|---|
| LSTM Encoder (Continuous Hidden State) | see metrics.json | see metrics.json | ~100K |
| GRU Encoder (Gated Continuous State) | see metrics.json | see metrics.json | ~85K |
| Transformer (Attention-Based Context) | see metrics.json | see metrics.json | ~95K |

**Output files:**
- `results/metrics.json` — Full results + autoresearch log
- `results/ade_fde_comparison.png` — Bar chart comparing ADE/FDE across architectures
- `results/training_curves.png` — Loss convergence per architecture
- `results/autoresearch_log.png` — ADE progression across autoresearch iterations

---

## Connection to the Paper

| Experiment Element | Paper Connection |
|---|---|
| LSTM latent encoder | Think2Drive (Section II-B) — RSSM latent planning |
| GRU latent encoder | AdaWM (Section II-B) — RNN-based latent state |
| Transformer encoder | LAW, DriveLaW (Section II-B) — attention-based latent |
| ADE metric | L2 Displacement Error (Section IV — Evaluation Standards) |
| Keep/Discard loop | Autoresearch template philosophy — one metric, automated iteration |
| Continuous latent form | Paper taxonomy — continuous vs. discrete vs. hybrid (Section II) |

---

## Paper Reference

Zeng, R., & Dong, Y. (2026). *Latent World Models for Automated Driving: A Unified Taxonomy, Evaluation Framework, and Open Challenges.* arXiv:2603.09086. IEEE TITS. https://arxiv.org/abs/2603.09086
