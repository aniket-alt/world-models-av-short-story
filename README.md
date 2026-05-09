# Short Story Assignment: Latent World Models for Autonomous Driving

**Course:** CMPE 258 - Deep Learning, San José State University  
**Name:** Aniket Anil Naik  
**Student ID:** 019107114

---

## 📄 Paper Details

**Title:** Latent World Models for Automated Driving: A Unified Taxonomy, Evaluation Framework, and Open Challenges  
**Authors:** Rongxiang Zeng (RWTH Aachen University) & Yongqi Dong (Delft University of Technology)  
**ArXiv:** [2603.09086](https://arxiv.org/abs/2603.09086)  
**Submitted:** March 10, 2026  
**Journal:** IEEE Transactions on Intelligent Transportation Systems (TITS)  
**Type:** Survey + Taxonomy + Evaluation Framework  
**License:** CC BY 4.0

---

## 📌 Deliverables

| Deliverable | Link |
|---|---|
| 📝 **Medium Article** | [Link coming soon] |
| 🖥️ **Slide Deck (SlideShare)** | [Link coming soon] |
| 🎥 **YouTube Video Presentation** | [Link coming soon] |
| 💻 **Code Reproduction** | See [`/code/reproduction/`](./code/reproduction/) |

---

## 📖 Paper Summary

Autonomous vehicles must deal with two brutal realities: the real world is dangerous and unpredictable, and you cannot crash a real car a million times to teach an AI to drive safely. The solution the research community is increasingly betting on is the **World Model** - an AI system that can *imagine the future* before it happens.

This 2026 survey paper, the first of its kind to take a **latent-centric lens** to the entire field, asks: across all the different world model approaches for autonomous driving, what are the shared design principles that determine success or failure?

### What is a Latent World Model?

A **Latent World Model (LWM)** is a world model that performs its "imagining" inside a compressed mathematical space called a *latent space*. Instead of predicting pixel-perfect video of the future, it compresses the driving scene into a small dense vector and predicts how that vector evolves over time. This is faster, more computationally efficient, and more amenable to planning and decision-making.

### The Paper's Core Contributions

**1. A Unified Taxonomy (4 Paradigms)**

| Paradigm | What It Imagines | Key Models |
|---|---|---|
| Spatiotemporal Neural Simulation | Future video / occupancy / point clouds | GAIA-2, DriveDreamer, BEVWorld, Epona, Orbis |
| Latent-Centric Planning & RL | Abstract latent state → driving action | Think2Drive, WorldRFT, LatentDriver, Raw2Drive |
| Generative Data Synthesis | Diverse synthetic training scenarios | MagicDrive-V2, DriveScape, CoGen, GLAD |
| Cognitive Reasoning (VLA) | Action + language explanation | DriveLaW, GenAD, BRYANT |

**2. Five Cross-Cutting Internal Mechanics**

These are the design principles that determine whether a latent world model actually works in real closed-loop deployment:

- **Structural Isomorphism & Geometric Priors** - the latent space must mirror real-world 3D geometry
- **Long-Horizon Temporal Stability** - errors must not compound over multi-second rollouts
- **Semantic & Reasoning Alignment** - latents must encode meaning, not just appearance
- **Value-Aligned Objectives & Post-Training** - the model must optimize for safety, not just reconstruction
- **Adaptive Computation & Deliberation** - spend more compute on hard decisions, less on easy ones

**3. A New Evaluation Framework**

The paper exposes a critical problem: models that score beautifully on open-loop metrics (FID, FVD) can still drive terribly in closed-loop. It proposes three new metrics:

- **Closed-loop Safety Gap (CSG)** - directly measures the gap between paper metrics and real deployment safety
- **Temporal Coherence Score (TCS)** - checks whether long-horizon rollouts remain physically plausible
- **Deliberation Cost (DC)** - resource-aware metric penalizing expensive models that aren't safer

---

## 📁 Repository Structure

```
world-models-av-short-story/
│
├── README.md                        ← You are here
│
├── paper/
│   └── notes.docx                   ← Phase 2 paper notes (full breakdown)
│
├── code/
│   └── reproduction/                ← Phase 3 autoresearch experiment
│       ├── experiment.ipynb         ← Main notebook
│       ├── train.py                 ← Training script
│       ├── results/
│       │   ├── metrics.json         ← Quantitative results
│       │   └── plots/               ← Visualizations
│       └── README.md                ← Explains what was reproduced and how
│
├── article/
│   └── medium_draft.md              ← Phase 4 Medium article draft
│
├── slides/
│   └── latent_world_models_slides.pdf ← Phase 4 slide deck
│
├── video/
│   └── README.md                    ← YouTube link + recording notes
│
└── assets/
    └── images/                      ← Diagrams and figures from the paper
```

---

## 🛠️ Code Reproduction

The code reproduction implements a **latent trajectory forecasting experiment** as a proxy for the core latent world model behavior described in the paper. Using the [`dlmastery/autoresearch`](https://github.com/dlmastery/autoresearch) template philosophy, the experiment:

1. Encodes vehicle trajectory sequences into a latent space using an LSTM encoder
2. Runs an autoresearch loop - systematically trying different latent architectures
3. Evaluates each architecture on trajectory prediction metrics (ADE, FDE, L2)
4. Reports the best-performing configuration with visualizations

**Dataset:** nuScenes mini / HighD trajectory data (lightweight, publicly available)  
**Metrics reproduced from paper:** L2 Displacement Error, Temporal Coherence

See [`/code/reproduction/README.md`](./code/reproduction/README.md) for full setup instructions.

---

## 🗺️ The Complete Mental Map

```
┌─────────────────────────────────────────────────────────────────────┐
│              LATENT WORLD MODEL - COMPLETE PICTURE                  │
├──────────────┬──────────────────────────────────────────────────────┤
│   INPUTS     │  Camera + LiDAR + HD Maps + Text + Ego State         │
├──────────────┼──────────────────────────────────────────────────────┤
│   COMPRESS   │  Raw Sensors → Latent Space                          │
│              │  (continuous vector / discrete tokens / hybrid)       │
├──────────────┼──────────────────────────────────────────────────────┤
│  4 PARADIGMS │  Neural Simulation | Latent RL Planning              │
│              │  Data Synthesis    | Cognitive Reasoning              │
├──────────────┼──────────────────────────────────────────────────────┤
│  5 MECHANICS │  Geometry | Stability | Semantics                     │
│              │  Value Alignment | Adaptive Compute                   │
├──────────────┼──────────────────────────────────────────────────────┤
│   OUTPUTS    │  Future Video / Occupancy / Action / Language         │
├──────────────┼──────────────────────────────────────────────────────┤
│  EVALUATION  │  FID/FVD (visual) | IoU (occupancy)                  │
│              │  Collision Rate/RC (closed-loop)                      │
│              │  CSG / TCS / DC (new - proposed in this paper)        │
└──────────────┴──────────────────────────────────────────────────────┘
```

---

## 📚 References

- **Primary Paper:** Zeng, R., & Dong, Y. (2026). *Latent World Models for Automated Driving: A Unified Taxonomy, Evaluation Framework, and Open Challenges.* arXiv:2603.09086. IEEE TITS. https://arxiv.org/abs/2603.09086
- **Background Survey:** Tu, S. et al. (2025/2026). *The Role of World Models in Shaping Autonomous Driving: A Comprehensive Survey.* arXiv:2502.10498v2.
- **Autoresearch Template:** dlmastery/autoresearch - https://github.com/dlmastery/autoresearch
- **Awesome World Model Repo:** LMD0311/Awesome-World-Model - https://github.com/LMD0311/Awesome-World-Model
