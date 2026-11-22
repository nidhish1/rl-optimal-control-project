# DreamNav

Simulated quadcopter navigation using a learned GRU world model and CEM-MPC, compared with a PPO baseline. Built with PyBullet, Gymnasium, PyTorch and Stable-Baselines3.

## Setup

Python 3.12 recommended. Run from the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

## Run

Trained models are included. Activate `.venv`, then run:

```bash
dreamnav demo --controller mpc --world runs/world/world.pt --gui
dreamnav demo --controller ppo --ppo runs/ppo/ppo.zip --gui
```

Compare both controllers on 30 held-out courses:

```bash
dreamnav evaluate --world runs/world/world.pt --ppo runs/ppo/ppo.zip --episodes 30
```

Use `--video output.mp4` instead of `--gui` to export a video. Keep the PPO checkpoint, `vecnormalize.pkl` and `training.json` together.

## Train

Raw flight data is generated locally. Collection requires a split with no existing episodes; use a new `--data` directory when needed.

```bash
dreamnav collect --split train --transitions 200000
dreamnav collect --split val --transitions 20000
dreamnav collect --split test --transitions 20000
dreamnav train-world --epochs 30
dreamnav train-ppo --transitions 200000
dreamnav eval-world --world runs/world/world.pt
```

Use `dreamnav --help` for commands and `dreamnav <command> --help` for options.

## Results and report

Both controllers completed 30/30 simple held-out missions without collisions. These courses reserve a clear route between waypoints; one training seed does not establish general obstacle-avoidance reliability.

- [Results and limitations](RESULTS.md)
- [Preliminary report (PDF)](output/pdf/preliminary_report.pdf) · [LaTeX](report/preliminary_report.tex) · [Overleaf ZIP](output/preliminary_report_latex.zip)
- Demo videos: [GRU-MPC](runs/demo-mpc/mpc.mp4) · [PPO](runs/demo-ppo/ppo.mp4)

Code is in `dreamnav/`, tests in `tests/`, and saved models/results in `runs/`. Raw data and `.venv` are excluded from Git. Transformer and occlusion experiments are future work.
