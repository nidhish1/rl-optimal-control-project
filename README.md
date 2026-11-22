# DreamNav

A simulated Crazyflie quadcopter learns flight dynamics from local sensing, plans with GRU + CEM-MPC, and is evaluated against PPO on identical courses. This implements the **core** of `project.pdf`. The Transformer and occlusion studies remain optional extensions.

Project repository: [nidhish1/rl-optimal-control-project](https://github.com/nidhish1/rl-optimal-control-project).

Preliminary report: [two-page PDF](output/pdf/preliminary_report.pdf) · [LaTeX source](report/preliminary_report.tex) · [Overleaf-ready ZIP](output/preliminary_report_latex.zip).

## Start here

Trained checkpoints and compact experiment results are included in this repository. Both learned controllers completed 30/30 held-out missions on the initial, forgiving courses. See [RESULTS.md](RESULTS.md) for evidence and limits. The original workspace already has a virtual environment; a fresh clone needs the installation commands below.

From this directory, launch the trained world-model controller:

```bash
source .venv/bin/activate
dreamnav demo --controller mpc --world runs/world/world.pt --gui
```

Ready-to-watch videos: [GRU + MPC](runs/demo-mpc/mpc.mp4), [PPO](runs/demo-ppo/ppo.mp4), and [scripted guidance](runs/demo/guided.mp4).

The guided controller is a scripted diagnostic, **not** the learned world model. It provides an immediate flight demonstration on a fresh installation before training. Export a video without opening a window:

```bash
dreamnav demo --controller guided --video runs/demo/guided.mp4
```

On a new machine, use Python 3.12 (tested) or 3.13:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

`gym-pybullet-drones` is pinned to a Git commit. `requirements-lock.txt` records the exact packages tested on macOS ARM64; other platforms may need different binary wheels. PyBullet may require a compiler if no wheel is available. Use `python -m dreamnav` in place of `dreamnav` if the console command is not on your PATH.

## Full experiment

Run these commands from the project root. Collection refuses to overwrite an existing split; choose a new `--data` directory for a new experiment.

```bash
# Generate independent layouts and complete episodes.
dreamnav collect --split train --transitions 200000
dreamnav collect --split val --transitions 20000
dreamnav collect --split test --transitions 20000

# Train and validate the predictive model; checkpoint selection uses validation only.
dreamnav train-world --epochs 30 --output runs/world
dreamnav eval-world --world runs/world/world.pt --output runs/forecasts

# Train PPO with the same sensors, commands, physics, rewards and mission specification.
dreamnav train-ppo --transitions 200000 --output runs/ppo

# Compare frozen models on the same 30 held-out courses.
dreamnav evaluate --world runs/world/world.pt --ppo runs/ppo/ppo.zip --episodes 30

# Show the learned controller's selected and alternative imagined trajectories.
dreamnav demo --controller mpc --world runs/world/world.pt --gui
dreamnav demo --controller mpc --world runs/world/world.pt --video runs/demo/mpc.mp4
dreamnav demo --controller ppo --ppo runs/ppo/ppo.zip --video runs/demo/ppo.mp4
```

Training budgets are starting points, not performance guarantees. Look at forecast errors, collision recall, actual navigation results, and latency before describing the system as successful. Selected checkpoints, videos, plots and metric files in `runs/` are included in Git; smoke runs and new experiment directories are ignored by default. Data manifests are included, while raw flight episodes and the virtual environment are excluded.

For a small pipeline check:

```bash
dreamnav smoke --output runs/smoke-new
```

This collects roughly 1,200 training transitions, trains a GRU for 3 epochs and PPO for 512 transitions, and runs both controllers. It deliberately does **not** test that either tiny model solves navigation.

## What is implemented

| Component | Implementation |
|---|---|
| Physics and stabilization | PyBullet CF2X, 240 Hz physics, 60 Hz PID |
| Navigation | 10 Hz, single drone, static 8 x 6 x 3 m rooms |
| Sensors | 32 body-fixed 3D range rays plus 12 motion/goal channels |
| Commands | Heading-frame planar velocity, world vertical velocity, Euler yaw rate |
| Data | Episode `.npz` files, collision/terminal labels, goal changes, layout IDs, sensor masks |
| World model | 44→128→64 encoder, 68-input GRU, hidden 128, residual observation and collision heads |
| Learning | Adam, masked episode windows, wrapped angle errors, train-only normalization, short rollout loss |
| Planning | Batched CEM, warm-started sequences, cloned recurrent history, invalid-forecast fallback |
| Baseline | Stable-Baselines3 PPO, separate 256×256 actor/critic MLPs, saved observation normalization |
| Evaluation | Shared layouts, success/collision rates, trajectory lengths, latency/deadline misses, forecast/persistence comparison |
| Demo | GUI trajectory overlay, headless MP4 export, labeled final frame |

There is no external dataset, camera-based perception, hardware deployment, motor-level RL, or Dreamer actor.

## Interfaces and correctness details

Observation indices are fixed:

| Slice | Meaning | Units/frame |
|---|---|---|
| `0:32` | Ray ranges | meters, body-fixed directions |
| `32:35` | Linear velocity | m/s, world |
| `35:38` | Roll/pitch/yaw | radians, wrapped to ±π |
| `38:41` | Angular velocity | rad/s, world |
| `41:44` | Current goal minus drone position | meters, world |

Actions lie in `[-1, 1]^4` and scale to ±0.8 m/s in heading-frame x/y, ±0.4 m/s vertically, and ±0.6 rad/s yaw rate. Heading-frame x/y means the horizontal frame rotates with yaw, rather than pitching with the drone. The command adapter integrates a yaw target and runs the PID at every control tick. These are **not** the stock `VelocityAviary` direction/speed actions.

Only observations are passed to the learned controllers. True obstacle geometry is never queried by MPC. The simulator's true goal position is used only to place predicted relative trajectories in the visualization.

On waypoint arrival, the environment returns the new goal observation to the controller and separately stores `model_next_observation` relative to the **old** goal. Sequence windows and GRU history reset at goal changes. This prevents learning a goal switch as a physical jump. The planner targets one waypoint at a time; mission management handles the sequence.

Layout IDs are disjoint by split: training `[0, 1000000)`, validation `[1000000, 2000000)`, test `[2000000, 3000000)`. Obstacle generation reserves a feasible tube along the waypoint route. These are useful controlled courses, not a claim of arbitrary maze-solving. Training collection mixes noisy goal guidance and bounded, temporally correlated random commands. Complete episodes can exceed the requested transition budget; the actual count is recorded.

The world-model loss combines normalized motion and masked range errors, weighted collision BCE, and a short recursive rollout loss. Collision class weighting improves training signal but means probabilities may require calibration; inspect Brier score and recall. The model starts near a persistence predictor. Low one-step error alone is not evidence that open-loop planning works.

MPC searches normalized action sequences. Its objective shares the environment's progress, collision, clearance, smoothness, time and goal terms, using predicted collision risk in place of actual collision events. It executes the first command and replans. Imagined candidates cannot modify the real-history GRU state. Nonfinite/implausible predictions or high immediate predicted collision risk trigger a zero-velocity/yaw command. This is a conservative simulation fallback, not a formal safety guarantee.

## Files and outputs

```text
dreamnav/
  config.py      Shared environment and planner settings
  env.py         Drone physics wrapper, sensors, rooms, rewards, rendering
  data.py        Exploration, episode storage and sequence datasets
  model.py       GRU dynamics and collision network
  train.py       World-model and PPO training
  planner.py     CEM-MPC with recurrent history
  evaluate.py    Shared evaluation, forecasts, plotting and video loop
  cli.py         Command-line workflows
tests/           Physics, interface, boundary and planning regression tests
data/flights/    Generated train/val/test episodes and manifests
runs/world/     world.pt, training.json, loss.png
runs/ppo/       ppo.zip, vecnormalize.pkl, training.json
runs/forecasts/ forecast.json, forecast.png, forecast_example.png
runs/evaluation/ summary.json, episodes.csv, trajectories.npz and plots
runs/demo/      Videos, final_frame.png, result.json, trajectory.npy
```

Keep `ppo.zip`, `vecnormalize.pkl`, and its `training.json` together. Load only checkpoints and normalization files you trust. Mean path/time metrics are calculated on successful episodes so early collisions are not rewarded as short paths. Latency excludes video rendering; simulator steps remain fixed even if a planner misses a real-time deadline, which is separately reported.

## Further experiments

Use `--scenario empty|sparse|dense`, `--waypoints 1|3|5|8`, and `--max-steps` for navigation experiments. Keep training and validation settings identical. Evaluation can change scenario and mission length while retaining sensor, command, reward and physics semantics.

```bash
dreamnav evaluate --world runs/world/world.pt --ppo runs/ppo/ppo.zip --horizon 20 --output runs/horizon20
dreamnav evaluate --world runs/world/world.pt --ppo runs/ppo/ppo.zip --waypoints 5 --output runs/five-waypoints
dreamnav evaluate --controllers guided mpc ppo --world runs/world/world.pt --ppo runs/ppo/ppo.zip --scenario dense --output runs/dense
```

Repeat training with `--seed 0`, `--seed 1`, and `--seed 2` in separate output directories for training-seed dispersion. Use the same evaluation seeds across controllers. Reserve test layouts for final reporting and use validation results for model/planner choices. The current CLI writes one checkpoint's results per evaluation; it does not claim a three-seed aggregate.

Use `evaluate --split val` for controller development; the default is `--split test`. `eval-world` also reports one-step collision diagnostics across **all** evaluation transitions so rare terminal collisions are not lost through forecast-window subsampling.

Forecast plots use the same 256 sampled 20-step windows at every horizon (or all eligible windows if fewer), sampled with a fixed seed across the split. Very short segments are excluded from those plots but included in the collision diagnostic.

## Upstream references

- [gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones), for drone simulation and PID.
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), for the model-free baseline.
- [Gymnasium environment integration](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html), for the shared training interface.
