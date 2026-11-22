"""Episode storage and sequence sampling with explicit waypoint boundaries."""
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import config_dict, layout_seed, save_json
from .env import DreamNavEnv, heading_rotation, lidar_directions


def guided_action(obs, config):
    """Local goal attraction + range repulsion; used only for data and diagnostics."""
    velocity = .65 * obs[41:44]
    norm = np.linalg.norm(velocity)
    if norm > .6:
        velocity *= .6 / norm
    # The rays are body-fixed; rotate their repulsion into the world frame.
    import pybullet as p
    rotation = np.asarray(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(obs[35:38]))).reshape(3, 3)
    rays = lidar_directions() @ rotation.T
    weight = np.maximum(0, .6 - obs[:32]) / .6
    velocity -= .35 * (rays * weight[:, None]).sum(axis=0)
    command = heading_rotation(obs[37]).T @ velocity
    return np.clip(np.r_[command / [config.max_speed_xy, config.max_speed_xy, config.max_speed_z], 0], -1, 1).astype(np.float32)


def collect(root, config, split="train", transitions=200_000, seed=0):
    if transitions < 1:
        raise ValueError("transitions must be positive")
    directory = Path(root) / split
    directory.mkdir(parents=True, exist_ok=True)
    if list(directory.glob("episode-*.npz")):
        raise ValueError(f"{directory} already contains episodes. Use a fresh data directory.")
    env = DreamNavEnv(config)
    rng = np.random.default_rng(seed)
    count = episode = collisions = 0
    started = time.perf_counter()
    try:
        while count < transitions:
            current_seed = layout_seed(split, seed + episode)
            obs, _ = env.reset(seed=current_seed)
            rows = {key: [] for key in ["observations", "actions", "next_observations", "collisions", "terminated",
                                        "truncated", "goal_changed", "goal_progress", "sensor_validity"]}
            noise = np.zeros(4)
            exploratory = rng.random() < .35
            for step in range(config.max_steps):
                if step % 8 == 0:
                    noise = rng.uniform(-1, 1, 4) * [1, 1, .65, .6]
                action = noise if exploratory else guided_action(obs, config) + .2 * noise
                action = np.clip(action, -1, 1).astype(np.float32)
                next_obs, _, terminated, truncated, info = env.step(action)
                values = [obs, action, info["model_next_observation"], info["collision"], terminated, truncated,
                          info["goal_changed"], info["goal_progress"], info["sensor_validity"]]
                for key, value in zip(rows, values):
                    rows[key].append(value)
                obs = next_obs
                if terminated or truncated:
                    break
            arrays = {key: np.asarray(value) for key, value in rows.items()}
            np.savez_compressed(directory / f"episode-{episode:06d}.npz", **arrays,
                                layout_id=np.array(current_seed), timestep=np.arange(len(rows["actions"])),
                                positions=np.asarray(env.trajectory), goals=env.goals)
            count += len(rows["actions"])
            collisions += int(info["collision"])
            episode += 1
            if episode % 10 == 0 or count >= transitions:
                print(f"{split}: {count} transitions, {episode} episodes, {collisions} collisions", flush=True)
    finally:
        env.close()
    manifest = {"schema_version": 1, "split": split, "env_config": config_dict(config), "seed": seed,
                "transitions": count, "requested_transitions": transitions, "episodes": episode,
                "collision_episodes": collisions, "seconds": time.perf_counter() - started,
                "note": "Complete episodes; count may exceed requested budget. Seeds are disjoint by split."}
    save_json(directory / "manifest.json", manifest)
    return manifest


def read_episodes(directory):
    paths = sorted(Path(directory).glob("episode-*.npz"))
    if not paths:
        raise ValueError(f"No episodes found in {directory}; run collect first.")
    episodes = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            episodes.append({key: archive[key] for key in archive.files})
    return episodes


def segments(episode):
    end = episode["goal_changed"] | episode["terminated"] | episode["truncated"]
    start = 0
    for index in np.flatnonzero(end):
        yield start, int(index) + 1
        start = int(index) + 1
    if start < len(end):
        yield start, len(end)


class EpisodeWindows(Dataset):
    def __init__(self, episodes, length=32):
        self.episodes = episodes
        self.length = length
        self.windows = []
        for ep_index, ep in enumerate(episodes):
            for begin, end in segments(ep):
                for start in range(begin, end, max(1, length // 2)):
                    self.windows.append((ep_index, start, min(end, start + length)))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        ep_index, start, end = self.windows[index]
        ep = self.episodes[ep_index]
        count = end - start
        result = {}
        for key in ["observations", "actions", "next_observations", "collisions", "sensor_validity"]:
            source = ep[key][start:end].astype(np.float32)
            padded = np.zeros((self.length, *source.shape[1:]), dtype=np.float32)
            padded[:count] = source
            result[key] = torch.from_numpy(padded)
        mask = torch.zeros(self.length)
        mask[:count] = 1
        result["mask"] = mask
        return result


def training_statistics(episodes):
    observations = np.concatenate([ep["observations"] for ep in episodes])
    mean = observations.mean(axis=0)
    std = np.maximum(observations.std(axis=0), .1)
    return mean.astype(np.float32), std.astype(np.float32)
