"""
=============================================================================
Phase 3 - Autoresearch Experiment
Paper : Latent World Models for Automated Driving (arXiv:2603.09086)
Course: CMPE 258 - Deep Learning, SJSU

WHAT THIS REPRODUCES:
    The paper's core claim is that latent representations compress
    high-dimensional observations and enable temporally coherent future
    state prediction (Section II-B: Latent-Centric Planning & RL).

    This script runs an autoresearch loop across three latent encoder
    architectures (LSTM, GRU, Transformer) on vehicle trajectory
    forecasting — directly demonstrating the latent prediction concept.

AUTORESEARCH LOOP:
    1. Define one metric : ADE (Average Displacement Error, meters)
    2. Define search space: {LSTM, GRU, Transformer} latent encoders
    3. For each architecture:
         a. Encode past trajectory → latent vector
         b. Decode latent → predicted future positions
         c. Evaluate ADE on held-out test set
         d. Keep if ADE improves, log everything
    4. Report best architecture + results table

METRIC:
    ADE (Average Displacement Error) — lower is better.
    Same as the L2 Displacement Error used in the paper (Section IV).

DATA:
    Synthetic trajectory data generated to match real driving statistics
    (nuScenes-style: 2s history → 3s future at 2Hz = 4 obs → 6 pred).
    Using synthetic data keeps this self-contained and reproducible
    on any machine (CPU-only, no downloads needed).
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# -- Reproducibility ------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -- Config ------------------------------------------------------------
OBS_LEN    = 8       # past timesteps (4 seconds at 2Hz)
PRED_LEN   = 12      # future timesteps (6 seconds at 2Hz)
INPUT_DIM  = 2       # (x, y) positions
LATENT_DIM = 64      # compressed latent vector size
HIDDEN_DIM = 128     # hidden layer size
N_SAMPLES  = 5000    # total trajectory samples
BATCH_SIZE = 64
EPOCHS     = 60
LR         = 1e-3
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"  Latent World Model — Autoresearch Experiment")
print(f"  Paper: arXiv:2603.09086")
print(f"  Device: {DEVICE}")
print(f"{'='*60}\n")


# ------------------------------------------------------------
#  DATA — Synthetic Driving Trajectories
#  Statistics match nuScenes (avg speed ~8 m/s, realistic curvature)
# ------------------------------------------------------------

def generate_driving_trajectories(n_samples, obs_len, pred_len, noise=0.05):
    """
    Generate synthetic vehicle trajectories mimicking real driving data.
    Three motion types: straight, left curve, right curve.
    This represents the kind of trajectory data used to train
    latent world models like Think2Drive and LatentDriver.
    """
    total_len = obs_len + pred_len
    trajs = []

    for _ in range(n_samples):
        motion_type = np.random.choice(['straight', 'curve_left', 'curve_right'],
                                        p=[0.5, 0.25, 0.25])
        speed  = np.random.uniform(3.0, 12.0)      # m/s (urban driving)
        dt     = 0.5                                # 2Hz sampling

        x, y   = 0.0, 0.0
        heading = np.random.uniform(-0.2, 0.2)     # slight initial heading variation
        traj    = []

        for t in range(total_len):
            if motion_type == 'curve_left':
                heading += np.random.uniform(0.02, 0.06)
            elif motion_type == 'curve_right':
                heading -= np.random.uniform(0.02, 0.06)

            x += speed * math.cos(heading) * dt + np.random.normal(0, noise)
            y += speed * math.sin(heading) * dt + np.random.normal(0, noise)
            traj.append([x, y])

        trajs.append(traj)

    trajs = np.array(trajs, dtype=np.float32)

    # Normalize to zero-mean, unit-variance per dimension
    mean = trajs.mean(axis=(0, 1))
    std  = trajs.std(axis=(0, 1)) + 1e-8
    trajs = (trajs - mean) / std

    obs  = trajs[:, :obs_len, :]
    pred = trajs[:, obs_len:, :]
    return obs, pred, mean, std


class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, obs, pred):
        self.obs  = torch.tensor(obs)
        self.pred = torch.tensor(pred)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.pred[idx]


# ------------------------------------------------------------
#  MODELS — Three Latent Encoder Architectures
#  Each represents a different latent representation form from the paper
# ------------------------------------------------------------

class LSTMLatentEncoder(nn.Module):
    """
    LSTM-based latent encoder.
    Represents: Continuous hidden state latent (used by Think2Drive, AdaWM).
    The hidden state h_t IS the latent representation of the driving scene.
    Paper connection: Section II-B — Latent-Centric Planning & RL.
    """
    def __init__(self, input_dim, latent_dim, hidden_dim, pred_len):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                               batch_first=True, dropout=0.1)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_len * input_dim)
        )
        self.pred_len   = pred_len
        self.input_dim  = input_dim
        self.latent_dim = latent_dim

    def encode(self, x):
        _, (h, _) = self.encoder(x)
        latent = self.latent_proj(h[-1])   # compress to latent space
        return latent

    def forward(self, x):
        latent = self.encode(x)
        out = self.decoder(latent)
        return out.view(-1, self.pred_len, self.input_dim), latent


class GRULatentEncoder(nn.Module):
    """
    GRU-based latent encoder.
    Represents: Gated continuous latent (used by DreamerV3-style models).
    GRU is lighter than LSTM — trades expressiveness for efficiency.
    Paper connection: Adaptive Computation mechanic (Section III-E).
    """
    def __init__(self, input_dim, latent_dim, hidden_dim, pred_len):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=2,
                              batch_first=True, dropout=0.1)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_len * input_dim)
        )
        self.pred_len  = pred_len
        self.input_dim = input_dim

    def encode(self, x):
        _, h = self.encoder(x)
        latent = self.latent_proj(h[-1])
        return latent

    def forward(self, x):
        latent = self.encode(x)
        out = self.decoder(latent)
        return out.view(-1, self.pred_len, self.input_dim), latent


class TransformerLatentEncoder(nn.Module):
    """
    Transformer-based latent encoder.
    Represents: Attention-based context latent (used by GenAD, LAW, DriveLaW).
    Uses self-attention to capture temporal dependencies in the observation.
    Paper connection: Semantic & Reasoning Alignment mechanic (Section III-C).
    """
    def __init__(self, input_dim, latent_dim, hidden_dim, pred_len, n_heads=4, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2, dropout=0.1,
            batch_first=True
        )
        self.transformer  = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.latent_proj  = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_len * input_dim)
        )
        self.pred_len  = pred_len
        self.input_dim = input_dim

    def encode(self, x):
        x_proj   = self.input_proj(x)
        context  = self.transformer(x_proj)
        pooled   = context.mean(dim=1)           # global average pool → latent
        latent   = self.latent_proj(pooled)
        return latent

    def forward(self, x):
        latent = self.encode(x)
        out    = self.decoder(latent)
        return out.view(-1, self.pred_len, self.input_dim), latent


# ------------------------------------------------------------
#  METRICS
# ------------------------------------------------------------

def compute_ade(pred, target):
    """
    Average Displacement Error - the paper's primary planning metric.
    Mean L2 distance between predicted and ground-truth positions
    across ALL future timesteps.
    Lower = better.
    """
    return torch.sqrt(((pred - target) ** 2).sum(dim=-1)).mean().item()


def compute_fde(pred, target):
    """
    Final Displacement Error - L2 distance at the last predicted timestep.
    Measures how accurately the model predicts the endpoint of the trajectory.
    Lower = better.
    """
    return torch.sqrt(((pred[:, -1, :] - target[:, -1, :]) ** 2).sum(dim=-1)).mean().item()


# ------------------------------------------------------------
#  TRAIN + EVALUATE
# ------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for obs, pred in loader:
        obs, pred = obs.to(DEVICE), pred.to(DEVICE)
        optimizer.zero_grad()
        pred_out, _ = model(obs)
        loss = criterion(pred_out, pred)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for obs, pred in loader:
            obs = obs.to(DEVICE)
            pred_out, _ = model(obs)
            all_preds.append(pred_out.cpu())
            all_targets.append(pred.cpu())
    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    ade = compute_ade(preds, targets)
    fde = compute_fde(preds, targets)
    return ade, fde


def run_experiment(name, model, train_loader, test_loader):
    """
    One iteration of the autoresearch loop.
    Train → evaluate → return ADE (the one metric we optimize).
    """
    print(f"\n-- Experiment: {name} ----------------------------")
    print(f"   Latent dim : {LATENT_DIM} | Hidden dim: {HIDDEN_DIM}")
    print(f"   Parameters : {sum(p.numel() for p in model.parameters()):,}")

    model     = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.MSELoss()

    train_losses = []
    start_time   = time.time()

    for epoch in tqdm(range(EPOCHS), desc=f"Training {name}", ncols=70):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        train_losses.append(loss)
        scheduler.step()

    elapsed = time.time() - start_time
    ade, fde = evaluate(model, test_loader)

    print(f"   ADE  : {ade:.4f}  (↓ lower is better)")
    print(f"   FDE  : {fde:.4f}  (↓ lower is better)")
    print(f"   Time : {elapsed:.1f}s")

    return {
        'name'         : name,
        'ade'          : round(ade, 4),
        'fde'          : round(fde, 4),
        'train_time_s' : round(elapsed, 1),
        'n_params'     : sum(p.numel() for p in model.parameters()),
        'train_losses' : train_losses
    }


# ------------------------------------------------------------
#  AUTORESEARCH LOOP
# ------------------------------------------------------------

def main():
    # -- Data ------------------------------------------------------------
    print("Generating synthetic driving trajectories...")
    obs, pred, mean, std = generate_driving_trajectories(
        N_SAMPLES, OBS_LEN, PRED_LEN)

    split       = int(0.8 * N_SAMPLES)
    train_ds    = TrajectoryDataset(obs[:split], pred[:split])
    test_ds     = TrajectoryDataset(obs[split:], pred[split:])
    train_loader = torch.utils.data.DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  BATCH_SIZE, shuffle=False)
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)} trajectories")

    # -- Architecture Search Space ------------------------------------------------------------
    experiments = [
        ("LSTM Encoder  (Continuous Hidden State)",
         LSTMLatentEncoder(INPUT_DIM, LATENT_DIM, HIDDEN_DIM, PRED_LEN)),
        ("GRU Encoder   (Gated Continuous State)",
         GRULatentEncoder(INPUT_DIM, LATENT_DIM, HIDDEN_DIM, PRED_LEN)),
        ("Transformer   (Attention-Based Context)",
         TransformerLatentEncoder(INPUT_DIM, LATENT_DIM, HIDDEN_DIM, PRED_LEN)),
    ]

    # -- Autoresearch Loop ------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  AUTORESEARCH LOOP — Optimizing ADE (lower = better)")
    print(f"  Search space: {len(experiments)} latent encoder architectures")
    print(f"{'='*60}")

    results     = []
    best_ade    = float('inf')
    best_name   = None
    log_entries = []

    for name, model in experiments:
        result = run_experiment(name, model, train_loader, test_loader)
        results.append(result)

        # -- Keep / Discard logic (autoresearch core) ------------------------------------------------------------
        improved = result['ade'] < best_ade
        if improved:
            prev_best = best_ade
            best_ade  = result['ade']
            best_name = name
            status    = f"✅ KEEP   (ADE {prev_best:.4f} → {best_ade:.4f})"
        else:
            status = f"❌ DISCARD (ADE {result['ade']:.4f} > best {best_ade:.4f})"

        log_entry = {
            'timestamp'  : datetime.now().isoformat(),
            'architecture': name,
            'ade'        : result['ade'],
            'fde'        : result['fde'],
            'decision'   : 'KEEP' if improved else 'DISCARD',
            'best_so_far': best_ade
        }
        log_entries.append(log_entry)
        print(f"\n  → Autoresearch decision: {status}")

    # -- Save Results ------------------------------------------------------------
    metrics_out = {
        'experiment'     : 'Latent Trajectory Forecasting — Autoresearch Loop',
        'paper'          : 'arXiv:2603.09086 — Latent World Models for Automated Driving',
        'metric_used'    : 'ADE (Average Displacement Error) — lower is better',
        'best_architecture': best_name,
        'best_ade'       : best_ade,
        'obs_len'        : OBS_LEN,
        'pred_len'       : PRED_LEN,
        'latent_dim'     : LATENT_DIM,
        'results'        : [
            {'architecture': r['name'], 'ade': r['ade'],
             'fde': r['fde'], 'n_params': r['n_params'],
             'train_time_s': r['train_time_s']}
            for r in results
        ],
        'autoresearch_log': log_entries
    }

    with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics_out, f, indent=2)

    # -- Plot 1: ADE Comparison Bar Chart ------------------------------------------------------------
    names   = [r['name'].split('(')[0].strip() for r in results]
    ades    = [r['ade'] for r in results]
    fdes    = [r['fde'] for r in results]
    colors  = ['#2E75B6' if r['name'] == best_name else '#A8C4E0' for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Latent World Model — Autoresearch Results\n(arXiv:2603.09086)',
                 fontsize=13, fontweight='bold', y=1.02)

    bars = axes[0].bar(names, ades, color=colors, edgecolor='white', width=0.5)
    axes[0].set_title('ADE by Latent Encoder Architecture', fontweight='bold')
    axes[0].set_ylabel('ADE (lower = better)')
    axes[0].set_ylim(0, max(ades) * 1.3)
    for bar, val in zip(bars, ades):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0].axhline(y=best_ade, color='green', linestyle='--', alpha=0.5,
                    label=f'Best ADE: {best_ade:.4f}')
    axes[0].legend(fontsize=9)
    axes[0].tick_params(axis='x', labelsize=9)

    bars2 = axes[1].bar(names, fdes, color=colors, edgecolor='white', width=0.5)
    axes[1].set_title('FDE by Latent Encoder Architecture', fontweight='bold')
    axes[1].set_ylabel('FDE (lower = better)')
    axes[1].set_ylim(0, max(fdes) * 1.3)
    for bar, val in zip(bars2, fdes):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1].tick_params(axis='x', labelsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'ade_fde_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # -- Plot 2: Training Loss Curves ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    line_styles = ['-', '--', '-.']
    colors_line = ['#1F4E79', '#2E75B6', '#70AD47']
    for i, r in enumerate(results):
        label = r['name'].split('(')[0].strip()
        ax.plot(r['train_losses'], label=label,
                linestyle=line_styles[i], color=colors_line[i], linewidth=2)
    ax.set_title('Training Loss Convergence — Autoresearch Iterations\n(arXiv:2603.09086)',
                 fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # -- Plot 3: Autoresearch Log (ADE over iterations) ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    iter_names = [e['architecture'].split('(')[0].strip() for e in log_entries]
    iter_ades  = [e['ade'] for e in log_entries]
    best_trail = [e['best_so_far'] for e in log_entries]

    ax.plot(iter_names, iter_ades, 'o-', color='#2E75B6',
            linewidth=2, markersize=8, label='ADE this iteration')
    ax.plot(iter_names, best_trail, 's--', color='green',
            linewidth=2, markersize=6, label='Best ADE so far')
    ax.fill_between(range(len(iter_names)), iter_ades, best_trail,
                    alpha=0.1, color='green')
    ax.set_title('Autoresearch Loop — ADE Progression\n(Keep/Discard decisions per iteration)',
                 fontweight='bold')
    ax.set_ylabel('ADE (lower = better)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'autoresearch_log.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # -- Final Summary ------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  AUTORESEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"\n  {'Architecture':<35} {'ADE':>8} {'FDE':>8} {'Params':>10}")
    print(f"  {'-'*65}")
    for r in sorted(results, key=lambda x: x['ade']):
        marker = ' ← BEST' if r['name'] == best_name else ''
        print(f"  {r['name']:<35} {r['ade']:>8.4f} {r['fde']:>8.4f} "
              f"{r['n_params']:>10,}{marker}")
    print(f"\n  Best architecture : {best_name}")
    print(f"  Best ADE          : {best_ade:.4f}")
    print(f"\n  Results saved to  : {RESULTS_DIR}/")
    print(f"    ├── metrics.json")
    print(f"    ├── ade_fde_comparison.png")
    print(f"    ├── training_curves.png")
    print(f"    └── autoresearch_log.png")
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
