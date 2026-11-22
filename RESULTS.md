# DreamNav: initial implementation results

The core prototype is implemented and trained. It includes PyBullet physics/PID, 32-ray local sensing, episode data generation, a GRU world model, CEM-MPC, PPO, matched evaluation, and video exports. The optional Transformer and memory/occlusion experiments are not implemented.

## Navigation

One trained checkpoint per learned controller was evaluated on the same 30 held-out sparse courses, layout IDs 2000000–2000029. Each mission had three waypoints. Model and planner choices were checked on five validation courses before this final comparison.

| Controller | Missions completed | Collisions | Mean completion time | Mean path length | Mean decision latency |
|---|---:|---:|---:|---:|---:|
| Scripted local guidance | 30 / 30 | 0 / 30 | 13.81 s | 7.03 m | 0.052 ms |
| GRU + CEM-MPC | 30 / 30 | 0 / 30 | 8.84 s | 7.67 m | 12.35 ms |
| PPO | 30 / 30 | 0 / 30 | 8.12 s | 8.01 m | 0.077 ms |

Flight times are simulation time. Latencies are local measurements, exclude rendering, and are not an isolated hardware benchmark. Navigation has a 100 ms decision budget. Actual per-episode latency percentiles and deadline misses are in the CSV.

Evidence: [summary](runs/evaluation/summary.json), [episode-level CSV](runs/evaluation/episodes.csv), [comparison chart](runs/evaluation/comparison.png), [trajectories](runs/evaluation/trajectories.png).

## Data and training

- World-model training: **200,056 transitions in 1,534 complete episodes**, including 495 terminal collision events. No external dataset.
- Validation: **20,049 transitions in 148 episodes**, disjoint layouts.
- Forecast test data: **20,010 transitions in 158 episodes**, disjoint layouts.
- GRU: 30 epochs, 32-step windows, five-step recursive training loss, Adam. Validation-selected checkpoint: epoch 24, weighted validation loss 0.2626. Fit time: approximately 85 seconds, excluding data collection.
- PPO: **200,704 simulator transitions**, four environments, approximately 318 seconds. The count rounds up to a complete PPO rollout.
- Both runs used training seed 0. These results do not include variation across independently trained seeds.

The GRU's extra validation simulation data and collection time are reported separately; equal training-transition budgets do not imply equal total compute or total simulator use. Training collection took approximately 283 seconds while other jobs also ran.

Evidence: [data manifest](data/flights/train/manifest.json), [GRU training record](runs/world/training.json), [PPO training record](runs/ppo/training.json), [loss curve](runs/world/loss.png).

## Prediction quality

All horizons use the same 256 sampled test windows, with up to 32 steps of real-history warm-up. No prediction crosses a waypoint change. Persistence means holding the current observation constant.

| Horizon | GRU normalized state RMSE | Persistence state RMSE | GRU LiDAR RMSE | Persistence LiDAR RMSE |
|---|---:|---:|---:|---:|
| 1 step / 0.1 s | 0.299 | 0.908 | 0.306 m | 0.367 m |
| 5 steps / 0.5 s | 0.404 | 0.966 | 0.403 m | 0.467 m |
| 10 steps / 1 s | 0.363 | 0.950 | 0.509 m | 0.631 m |
| 20 steps / 2 s | 0.466 | 0.995 | 0.611 m | 0.864 m |

One-step collision estimates over all 20,010 test transitions had **88.4% recall and 40.9% precision** at probability threshold 0.5, with 43 positive events. Class-weighted training does not ensure calibrated probabilities. The sampled long-horizon forecasts contained very few collision events and missed the single positive event at 20 steps, so they do not establish reliable long-range collision prediction.

Evidence: [forecast metrics](runs/forecasts/forecast.json), [forecast comparison](runs/forecasts/forecast.png), [example rollout](runs/forecasts/forecast_example.png).

## Verification and demonstrations

The automated tests cover the Gymnasium interface, reproducible resets, hover/velocity/yaw control, collisions, LiDAR, waypoint target handling, time limits, independent simulator clients, split IDs, sequence boundaries, padding masks, angle wrapping, recurrent planning history, invalid-prediction fallback, and matched forecast populations.

The end-to-end smoke run exercised collection, GRU training, PPO training, checkpoint loading, prediction evaluation and navigation evaluation. Its tiny models intentionally do not establish task performance.

- [GRU + MPC video](runs/demo-mpc/mpc.mp4): three waypoints, no collision, 8.7 seconds of simulated flight. Orange/blue paths are learned forecasts; green is the actual path.
- [PPO video](runs/demo-ppo/ppo.mp4): the same course, three waypoints, no collision.
- [Scripted diagnostic video](runs/demo/guided.mp4): local guidance with no learned model.

The videos were decoded and frames/plots inspected after export. See the [README](README.md) for launch and reproduction commands.

## Limits and next experiments

These initial courses are intentionally forgiving: fixed room size and starting location, modest waypoint jitter, and a reserved collision-free tube along the waypoint route. The scripted controller also succeeds on every course. The result demonstrates a functioning learned-prediction control pipeline, **not superiority over PPO or general obstacle-avoidance reliability**.

The next scientific step is a separate challenge set with obstacles blocking direct waypoint paths, varied room widths/start positions, and narrow passages, followed by multiple independent training seeds. More diverse on-policy world-model data and collision calibration should precede longer horizons or moving obstacles. Longer missions and denser layouts are exposed by CLI flags but were not benchmarked in this initial run.
