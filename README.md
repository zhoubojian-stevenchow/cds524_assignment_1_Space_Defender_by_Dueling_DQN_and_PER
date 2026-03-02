# 🚀 Space Defender: Q-Learning Game Design with Dueling DQN and PER

**CDS524 Assignment 1 — Reinforcement Learning Game Design**

Zhou Bojian · MSc AI and Business Analytics · Lingnan University, Hong Kong · March 2026

---

## Overview

Space Defender is a vertical-scrolling shooter game where an autonomous agent learns to navigate a hostile environment, eliminate enemies, and survive as long as possible — entirely through reinforcement learning. The project demonstrates an iterative development journey from a basic DQN (v1) to a state-of-the-art **Dueling DQN with Prioritized Experience Replay** (v7), achieving consistent scores of **3500 ± 300**.

The agent controls a spaceship that moves horizontally and shoots vertically to destroy incoming enemy ships while dodging enemy bullets. Over 7 development iterations, the project addresses challenges including reward hacking, training instability, and sample inefficiency, culminating in a clean and effective architecture.

## Game Design

**Objective:** Destroy as many enemy ships as possible while surviving. The player ship has 3 health points and loses one when hit by an enemy bullet or upon enemy collision.

**State Space:** A 22-dimensional feature vector per frame (player position, shooting availability, health status, relative positions and threat levels of the 3 nearest enemies and 3 nearest bullets, plus global counts) stacked across 4 consecutive frames for temporal context, yielding an **88-dimensional** input.

**Action Space:** 6 discrete actions — idle, move left, move right, shoot, move left + shoot, move right + shoot.

**Reward Structure (v7 — Simplified):**

| Signal | Reward |
|--------|--------|
| Survive per frame | +0.01 |
| Hit an enemy | +10 |
| Kill an enemy | +50 |
| Take damage | −30 |
| Die | −100 |

A key insight from the iterative development was that **complex reward shaping leads to reward hacking**. The v7 minimalist design forces the agent to discover effective strategies organically.

## Architecture

### Dueling DQN

The network separates the state value **V(s)** from the action advantage **A(s, a)**, combining them as:

```
Q(s, a) = V(s) + (A(s, a) − mean(A(s, :)))
```

**Network structure (~430K parameters):**
- Shared feature extraction: 88 → 512 → 512 → 256 (ReLU, orthogonal initialization)
- Value stream: 256 → 128 → 1
- Advantage stream: 256 → 128 → 6

### Prioritized Experience Replay (PER)

Implemented via a **SumTree** data structure for O(log n) proportional sampling. Transitions with higher TD-error (surprising outcomes, deaths, kills) are replayed more frequently, with importance-sampling weights to correct for bias.

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate (α) | 0.0001 |
| Discount factor (γ) | 0.99 |
| Epsilon decay | 1.0 → 0.02, factor 0.9995 |
| Batch size | 128 |
| Replay buffer size | 200,000 |
| Target network update | Every 500 steps |
| PER α | 0.6 |
| PER β | 0.4 → 1.0 |

## Version History

This repository documents the full iterative development process across 7 versions:

| Version | File | Key Focus |
|---------|------|-----------|
| **v1** | `SpaceDefender_QLearning_Colab_v1.ipynb` | Baseline DQN with standard replay memory. Initial training from scratch (10,000 episodes). |
| **v2** | `SpaceDefender_FineTune_ProMode_v2.ipynb` | Fine-tuning to fix corner-hiding behavior. Added edge/corner penalties, anti-camping penalties, and center position bonuses. |
| **v3** | `SpaceDefender_FineTune_ProMode_v3_average.ipynb` | Continued fine-tuning with adjusted reward shaping to improve average performance. |
| **v4** | `SpaceDefender_FineTune_ProMode_v4.ipynb` | Further reward tuning with higher kill rewards (+50) and lower death penalties (−200). |
| **v5** | `SpaceDefender_FineTune_ProMode_v5_FlexiblePosition.ipynb` | Flexible positioning — removed rigid center bonus, added dynamic evasion rewards to fix "stuck in center getting surrounded" behavior. |
| **v6** | `SpaceDefender_v6_StableTraining.ipynb` | **Dueling DQN** introduced. Soft target updates (τ=0.005), lower learning rate (1e-5), knowledge transfer from v4 checkpoint. |
| **v7** | `SpaceDefender_v7_SimplifiedRewards_PER.ipynb` | **Final version.** Clean-slate training with simplified 5-signal rewards, Dueling DQN + PER. Target: 3500 ± 300 consistent score. |

## Getting Started

### Prerequisites

- Python 3.8+
- Google Colab (recommended, all notebooks are Colab-ready)
- GPU runtime (A100 recommended for faster training)

### Dependencies

```bash
pip install pygame numpy torch matplotlib imageio pyvirtualdisplay
apt-get install -y xvfb ffmpeg
```

### Running

1. Open any `.ipynb` file in Google Colab.
2. Mount your Google Drive (Step 1 in each notebook).
3. Run all cells sequentially.
4. Training outputs (checkpoints, logs, plots, videos) are saved to your Google Drive.

**To run the final version (recommended):** Open `SpaceDefender_v7_SimplifiedRewards_PER.ipynb` and execute all cells. Training runs for 10,000 episodes with evaluations every 5 episodes (30 games each).

## Results

The v7 agent learns emergent behaviors including strategic positioning, enemy targeting prioritization, and bullet dodging — all discovered through simplified rewards rather than explicit reward engineering. Key results:

- **Consistent scores of 3500+** after training
- Evaluation uses 30-game averages for statistical reliability
- 100-episode warmup period for initial experience collection
- Training metrics tracked: mean score, standard deviation, kills per game, survival time

## Challenges & Solutions

**Reward Hacking (v1–v4):** Complex reward shaping with 15+ signals caused the agent to exploit intermediate rewards instead of learning intended behaviors. → *Solved in v7 with a minimalist 5-signal reward structure.*

**Training Instability (v1–v5):** Standard DQN showed high variance with good policies degrading during training. → *Solved with Dueling DQN, Double DQN action selection, and gradient clipping (max_norm=10.0).*

**Sample Inefficiency:** Critical transitions (deaths, kills) were rare compared to normal gameplay. → *Solved with Prioritized Experience Replay, which upsamples high-TD-error transitions.*

## Report

The full technical report is available in [`SpaceDefender_Report.docx`](SpaceDefender_Report.docx), covering game design, algorithm implementation, system architecture, evaluation results, and challenges in detail.

## References

1. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529–533.
2. Wang, Z., et al. (2016). Dueling Network Architectures for Deep Reinforcement Learning. *ICML*.
3. Schaul, T., et al. (2016). Prioritized Experience Replay. *ICLR*.
4. Van Hasselt, H., et al. (2016). Deep Reinforcement Learning with Double Q-learning. *AAAI*.

## License

This project is developed for academic purposes as part of CDS524 at Lingnan University.
