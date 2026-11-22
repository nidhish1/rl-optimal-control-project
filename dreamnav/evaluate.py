import csv
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .config import EnvConfig, config_dict, layout_seed, save_json
from .data import guided_action, read_episodes, segments
from .env import DreamNavEnv
from .model import load_world_model, prediction_difference
from .planner import CEMPlanner


def plotting():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/dreamnav-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def validate_config(saved, config):
    # Vary missions, not sensor, action, physics or reward definitions.
    for key, value in config_dict(config).items():
        if key not in {"scenario", "waypoints", "max_steps"} and saved[key] != value:
            raise ValueError(f"Checkpoint/environment mismatch for {key}: {saved[key]} != {value}")


class Controller:
    def __init__(self, name, config, checkpoint=None, mpc_config=None, seed=0):
        self.name, self.config = name, config
        self.planner = self.normalizer = None
        self.training_transitions = None
        if name == "mpc":
            if checkpoint is None:
                raise ValueError("MPC requires --world checkpoint.pt")
            model, metadata = load_world_model(checkpoint)
            validate_config(metadata["env_config"], config)
            self.planner = CEMPlanner(model, config, mpc_config, seed)
            self.training_transitions = metadata["train_transitions"]
        elif name == "ppo":
            if checkpoint is None:
                raise ValueError("PPO requires --ppo path/to/ppo.zip")
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            checkpoint = Path(checkpoint)
            metadata = json.loads((checkpoint.parent / "training.json").read_text())
            validate_config(metadata["env_config"], config)
            self.training_transitions = metadata["train_transitions"]
            dummy = DummyVecEnv([lambda: DreamNavEnv(config)])
            self.normalizer = VecNormalize.load(checkpoint.parent / "vecnormalize.pkl", dummy)
            self.normalizer.training = False
            self.normalizer.norm_reward = False
            self.ppo = PPO.load(checkpoint, device="cpu")
        elif name not in {"guided", "hover"}:
            raise ValueError(f"Unknown controller {name}")

    def reset(self):
        if self.planner:
            self.planner.reset()

    def act(self, obs):
        if self.planner:
            return self.planner.act(obs)
        if self.name == "ppo":
            normalized = self.normalizer.normalize_obs(obs[None])
            return self.ppo.predict(normalized, deterministic=True)[0][0]
        if self.name == "hover":
            return np.zeros(4, dtype=np.float32)
        return guided_action(obs, self.config)

    def update(self, obs, action, info):
        if self.planner:
            self.planner.observe_transition(obs, action, info["goal_changed"])

    def close(self):
        if self.normalizer:
            self.normalizer.close()


def run_episode(env, controller, seed, video=None, realtime=False):
    import imageio.v2 as imageio
    obs, info = env.reset(seed=seed)
    env.controller_label = {"guided": "Scripted guidance", "mpc": "Learned GRU + MPC", "ppo": "PPO policy", "hover": "Hover check"}[controller.name]
    controller.reset()
    latencies = []
    rewards = 0.
    writer = imageio.get_writer(str(video), fps=env.config.navigation_hz) if video else None
    if writer:
        writer.append_data(env.render())
    try:
        for _ in range(env.config.max_steps):
            tick = time.perf_counter()
            action = controller.act(obs)
            latencies.append((time.perf_counter() - tick) * 1000)
            if controller.planner:
                predictions = controller.planner.last_predictions
                if predictions is not None:
                    # True goal position is used only for visualization.
                    goal = env.goals[env.goal_index]
                    paths = goal - predictions[..., 41:44]
                    starts = np.broadcast_to(info["position"], (len(paths), 1, 3))
                    env.set_candidate_paths(np.concatenate([starts, paths], axis=1))
                else:
                    env.set_candidate_paths([])
            next_obs, reward, terminated, truncated, info = env.step(action)
            controller.update(obs, action, info)
            obs = next_obs
            rewards += reward
            if writer:
                writer.append_data(env.render())
            if realtime:
                time.sleep(max(0, 1 / env.config.navigation_hz - (time.perf_counter() - tick)))
            if terminated or truncated:
                break
    finally:
        if writer:
            writer.close()
    row = {"controller": controller.name, "layout_id": seed, "success": bool(info["is_success"]),
           "collision": bool(info["collision"]), "timeout": bool(truncated),
           "waypoints_reached": info["waypoints_reached"], "path_length_m": info["path_length"],
           "duration_s": info["elapsed_seconds"], "return": rewards,
           "latency_mean_ms": float(np.mean(latencies)), "latency_p95_ms": float(np.percentile(latencies, 95)),
           "deadline_miss_fraction": float(np.mean(np.array(latencies) > 1000 / env.config.navigation_hz)),
           "fallbacks": controller.planner.fallbacks if controller.planner else 0,
           "training_transitions": controller.training_transitions}
    return row, np.asarray(env.trajectory)


def evaluate(config, output, controllers, world=None, ppo=None, mpc_config=None, episodes=30, seed=0, split="test"):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = DreamNavEnv(config)
    rows, paths, scene = [], {}, None
    try:
        for name in controllers:
            controller = Controller(name, config, world if name == "mpc" else ppo, mpc_config, seed)
            try:
                for episode in range(episodes):
                    row, path = run_episode(env, controller, layout_seed(split, seed + episode))
                    rows.append(row)
                    paths[f"{name}_{episode}"] = path
                    if scene is None:
                        scene = {"layout_id": row["layout_id"], "goals": env.goals.tolist(),
                                 "obstacles": [{"center": center.tolist(), "half_extents": half.tolist()}
                                               for center, half in env.obstacles]}
                    print(json.dumps(row), flush=True)
            finally:
                controller.close()
    finally:
        env.close()
    with (output / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "trajectories.npz", **paths)
    save_json(output / "first_course.json", scene)
    summaries = {}
    for name in controllers:
        subset = [row for row in rows if row["controller"] == name]
        successes = [row for row in subset if row["success"]]
        summaries[name] = {"episodes": len(subset), "success_rate": float(np.mean([r["success"] for r in subset])),
                           "collision_rate": float(np.mean([r["collision"] for r in subset])),
                           "mean_waypoints": float(np.mean([r["waypoints_reached"] for r in subset])),
                           "mean_latency_ms": float(np.mean([r["latency_mean_ms"] for r in subset])),
                           "mean_path_length_success_m": float(np.mean([r["path_length_m"] for r in successes])) if successes else None,
                           "mean_duration_success_s": float(np.mean([r["duration_s"] for r in successes])) if successes else None,
                           "training_transitions": subset[0]["training_transitions"]}
    result = {"env_config": config_dict(config), "mpc_config": config_dict(mpc_config) if mpc_config else None,
              "split": split, "layout_seed_start": layout_seed(split, seed), "controllers": summaries,
              "note": "One checkpoint/training seed per controller; repeat for training-seed dispersion. Timing excludes rendering."}
    save_json(output / "summary.json", result)
    plt = plotting()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, metric, label in zip(axes, ["success_rate", "collision_rate", "mean_latency_ms"],
                                ["Mission success rate", "Collision rate", "Mean decision latency (ms)"]):
        ax.bar(controllers, [summaries[n][metric] for n in controllers], color=["#228be6", "#12b886", "#fab005", "#868e96"][:len(controllers)])
        ax.set_title(label)
        if metric.endswith("rate"):
            ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    from matplotlib.patches import Rectangle
    for obstacle in scene["obstacles"]:
        center, half = np.asarray(obstacle["center"]), np.asarray(obstacle["half_extents"])
        ax.add_patch(Rectangle(center[:2] - half[:2], 2 * half[0], 2 * half[1], facecolor="#cbd5e1", edgecolor="#64748b"))
    goals = np.asarray(scene["goals"])
    ax.scatter(goals[:, 0], goals[:, 1], marker="*", s=100, color="#087f5b", label="Waypoints", zorder=5)
    for name in controllers:
        path = paths[f"{name}_0"]
        ax.plot(path[:, 0], path[:, 1], label=name)
    ax.set(xlabel="World x (m)", ylabel="World y (m)", title="First held-out course: top view", xlim=(-4, 4), ylim=(-3, 3))
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output / "trajectories.png", dpi=150)
    plt.close(fig)
    return result


@torch.inference_mode()
def evaluate_world(checkpoint, data_dir, output, split="test", max_windows=256):
    if split == "train":
        raise ValueError("Forecast evaluation requires val or test data.")
    if max_windows < 1:
        raise ValueError("max_windows must be positive")
    torch.set_num_threads(1)
    model, metadata = load_world_model(checkpoint)
    manifest = json.loads((Path(data_dir) / split / "manifest.json").read_text())
    validate_config(metadata["env_config"], EnvConfig(**manifest["env_config"]))
    episodes = read_episodes(Path(data_dir) / split)
    ids = {int(ep["layout_id"]) for ep in episodes}
    if ids & set(metadata["train_layout_ids"]):
        raise ValueError("Forecast evaluation data overlap the training layouts.")
    if split == "test" and ids & set(metadata["val_layout_ids"]):
        raise ValueError("Test data overlap validation layouts used for checkpoint selection.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = {h: [] for h in [1, 5, 10, 20]}
    example = None
    candidates = [(i, begin, start) for i, ep in enumerate(episodes)
                  for begin, end in segments(ep) for start in range(begin, end - 19, 20)]
    if not candidates:
        raise ValueError("Forecast evaluation needs at least one 20-step segment without a goal switch.")
    chosen = np.random.default_rng(0).permutation(len(candidates))[:max_windows]
    windows = len(chosen)
    # Use matched windows to avoid survivor bias across horizons.
    for index in chosen:
        ep_index, begin, start = candidates[index]
        ep = episodes[ep_index]
        obs = torch.as_tensor(ep["observations"], dtype=torch.float32)
        actions = torch.as_tensor(ep["actions"], dtype=torch.float32)
        target = torch.as_tensor(ep["next_observations"], dtype=torch.float32)
        hidden = None
        if start > begin:
            _, _, hidden = model(obs[None, max(begin, start - 32):start], actions[None, max(begin, start - 32):start])
        current = obs[start:start + 1]
        predicted = []
        for k in range(20):
            current, logits, hidden = model.step(current, actions[start + k:start + k + 1], hidden)
            predicted.append(current[0].numpy())
            if k + 1 in records:
                truth = target[start + k]
                error = prediction_difference(current[0], truth)
                persistence = prediction_difference(obs[start], truth)
                valid_rays = torch.as_tensor(ep["sensor_validity"][start + k], dtype=torch.bool)
                lidar_mse = float(error[:32][valid_rays].square().mean()) if valid_rays.any() else None
                records[k + 1].append({"state_normalized_mse": float((error[32:] / model.std[32:]).square().mean()),
                                       "lidar_mse": lidar_mse,
                                       "persistence_state_normalized_mse": float((persistence[32:] / model.std[32:]).square().mean()),
                                       "persistence_lidar_mse": float(persistence[:32][valid_rays].square().mean()) if valid_rays.any() else None,
                                       "collision_probability": float(torch.sigmoid(logits[0])),
                                       "collision_label": float(ep["collisions"][start + k])})
        if example is None:
            example = (np.asarray(predicted), target[start:start + 20].numpy())
    summary = {}
    for horizon, values in records.items():
        if not values:
            continue
        probs = np.array([v["collision_probability"] for v in values])
        labels = np.array([v["collision_label"] for v in values])
        summary[str(horizon)] = {"samples": len(values), "collision_positives": int(labels.sum()),
                                 "collision_brier_score": float(np.square(probs - labels).mean()),
                                 "collision_recall_at_0.5": float(((probs >= .5) & (labels == 1)).sum() / labels.sum()) if labels.sum() else None}
        for key in ["state_normalized_mse", "lidar_mse", "persistence_state_normalized_mse", "persistence_lidar_mse"]:
            numbers = [v[key] for v in values if v[key] is not None]
            summary[str(horizon)][key.replace("mse", "rmse")] = float(np.sqrt(np.mean(numbers))) if numbers else None
    # Count all collisions, including those outside sampled windows.
    all_probs, all_labels = [], []
    for ep in episodes:
        for begin, end in segments(ep):
            hidden = None
            for start in range(begin, end, 128):
                stop = min(start + 128, end)
                _, logits, hidden = model(torch.tensor(ep["observations"][None, start:stop], dtype=torch.float32),
                                          torch.tensor(ep["actions"][None, start:stop], dtype=torch.float32), hidden)
                all_probs.extend(torch.sigmoid(logits).flatten().tolist())
                all_labels.extend(ep["collisions"][start:stop].tolist())
    probs, labels = np.asarray(all_probs), np.asarray(all_labels, dtype=bool)
    predicted_positive = probs >= .5
    tp = int((predicted_positive & labels).sum())
    diagnostics = {"transitions": len(labels), "positive_transitions": int(labels.sum()),
                   "brier_score": float(np.square(probs - labels).mean()),
                   "recall_at_0.5": tp / int(labels.sum()) if labels.any() else None,
                   "precision_at_0.5": tp / int(predicted_positive.sum()) if predicted_positive.any() else None}
    save_json(output / "forecast.json", {"split": split, "windows": windows, "horizons": summary,
                                          "sampling": "Fixed seed 0; same uniformly sampled 20-step windows at every horizon",
                                          "one_step_collision_all_transitions": diagnostics})
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    hs = [int(h) for h in summary]
    for ax, key, title in zip(axes, ["state_normalized_rmse", "lidar_rmse"], ["State error (normalized)", "LiDAR error (m)"]):
        ax.plot(hs, [summary[str(h)][key] for h in hs], "o-", label="GRU")
        ax.plot(hs, [summary[str(h)]["persistence_" + key] for h in hs], "o--", label="Persistence")
        ax.set(xlabel="Open-loop steps", ylabel="RMSE", title=title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "forecast.png", dpi=150)
    plt.close(fig)
    if example:
        prediction, truth = example
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(truth[:, 41], label="Actual")
        axes[0].plot(prediction[:, 41], label="Predicted")
        axes[0].set_title("Goal displacement x (m)")
        axes[1].plot(truth[:, :32].mean(-1), label="Actual")
        axes[1].plot(prediction[:, :32].mean(-1), label="Predicted")
        axes[1].set_title("Mean LiDAR range (m)")
        for ax in axes:
            ax.set_xlabel("Open-loop step")
            ax.legend()
        fig.tight_layout()
        fig.savefig(output / "forecast_example.png", dpi=150)
        plt.close(fig)
    return summary
