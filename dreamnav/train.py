import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import config_dict, layout_seed, save_json
from .data import EpisodeWindows, read_episodes, training_statistics
from .model import GRUWorldModel, prediction_difference


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def batch_loss(model, batch, positive_weight, rollout_steps=0):
    obs, actions, target, mask = (batch[k] for k in ["observations", "actions", "next_observations", "mask"])
    prediction, collision, _ = model(obs, actions)
    error = prediction_difference(prediction, target) / model.std
    valid_rays = batch["sensor_validity"] * mask[..., None]
    lidar = (error[..., :32].square() * valid_rays).sum() / valid_rays.sum().clamp_min(1)
    state = (error[..., 32:].square().mean(-1) * mask).sum() / mask.sum().clamp_min(1)
    collision_loss = F.binary_cross_entropy_with_logits(collision, batch["collisions"], reduction="none", pos_weight=positive_weight)
    collision_loss = (collision_loss * mask).sum() / mask.sum().clamp_min(1)
    loss = lidar + state + .2 * collision_loss
    if rollout_steps:
        # Short open-loop training from the same segment, preceded by real-history burn-in.
        burn = min(5, obs.shape[1] - 1)
        _, _, hidden = model(obs[:, :burn], actions[:, :burn]) if burn else (None, None, None)
        current = obs[:, burn]
        rollout_loss = torch.zeros((), device=obs.device)
        count = min(rollout_steps, obs.shape[1] - burn)
        for k in range(count):
            t = burn + k
            current, _, hidden = model.step(current, actions[:, t], hidden)
            err = prediction_difference(current, target[:, t]) / model.std
            channel_mask = torch.cat([batch["sensor_validity"][:, t], torch.ones_like(err[:, 32:])], -1)
            valid = channel_mask * mask[:, t, None]
            rollout_loss += (err.square() * valid).sum() / valid.sum().clamp_min(1)
        loss += .2 * rollout_loss / max(count, 1)
    return loss


def train_world(data_dir, output, epochs=30, batch_size=64, length=32, seed=0, device="cpu", rollout_steps=5):
    if epochs < 1 or batch_size < 1 or length < 2 or rollout_steps < 0:
        raise ValueError("Use positive epochs/batch size, sequence length >= 2 and rollout_steps >= 0.")
    seed_everything(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    training = read_episodes(Path(data_dir) / "train")
    validation = read_episodes(Path(data_dir) / "val")
    train_ids = {int(ep["layout_id"]) for ep in training}
    val_ids = {int(ep["layout_id"]) for ep in validation}
    if train_ids & val_ids:
        raise ValueError("Training and validation layouts overlap.")
    manifests = [json.loads((Path(data_dir) / split / "manifest.json").read_text()) for split in ["train", "val"]]
    if manifests[0]["env_config"] != manifests[1]["env_config"]:
        raise ValueError("Training and validation environment configurations differ.")
    mean, std = training_statistics(training)
    model = GRUWorldModel(mean, std).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    train_loader = DataLoader(EpisodeWindows(training, length), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(EpisodeWindows(validation, length), batch_size=batch_size)
    positives = sum(ep["collisions"].sum() for ep in training)
    total = sum(len(ep["actions"]) for ep in training)
    pos_weight = torch.tensor(min(20., max(1., (total - positives) / max(1, positives))), device=device)
    history, best = [], float("inf")
    started = time.perf_counter()
    for epoch in range(epochs):
        losses = []
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = batch_loss(model, batch, pos_weight, rollout_steps)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite world-model loss; inspect data/normalization.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.inference_mode():
            val_losses = [batch_loss(model, {k: v.to(device) for k, v in batch.items()}, pos_weight, rollout_steps).item()
                          for batch in val_loader]
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": float(np.mean(val_losses))}
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["val_loss"] < best:
            best = row["val_loss"]
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val_loss": best,
                        "env_config": manifests[0]["env_config"], "train_transitions": total,
                        "train_layout_ids": sorted(train_ids), "val_layout_ids": sorted(val_ids),
                        "seed": seed, "architecture": "gru", "schema_version": 1}, output / "world.pt")
    result = {"history": history, "seconds": time.perf_counter() - started, "train_transitions": total,
              "validation_transitions": sum(len(ep["actions"]) for ep in validation),
              "collision_positive_transitions": int(positives), "positive_weight": float(pos_weight),
              "seed": seed, "sequence_length": length, "rollout_training_steps": rollout_steps,
              "best_val_loss": best}
    save_json(output / "training.json", result)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/dreamnav-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([x["train_loss"] for x in history], label="Train")
    ax.plot([x["val_loss"] for x in history], label="Validation")
    ax.set(xlabel="Epoch (zero-indexed)", ylabel="Weighted prediction loss", title="GRU world model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "loss.png", dpi=150)
    plt.close(fig)
    return result


def train_ppo(config, output, transitions=200_000, seed=0, n_envs=4):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from .env import DreamNavEnv

    if transitions < 1 or n_envs < 1:
        raise ValueError("transitions and n_envs must be positive")
    seed_everything(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    class TrainingLayouts(DreamNavEnv):
        def __init__(self, index):
            super().__init__(config)
            self.layout_rng = np.random.default_rng(seed * 1000 + index)

        def reset(self, *, seed=None, options=None):
            return super().reset(seed=layout_seed("train", int(self.layout_rng.integers(1_000_000))), options=options)

    env = DummyVecEnv([lambda index=i: Monitor(TrainingLayouts(index)) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    # Small budgets are supported for pipeline checks; the actual rounded count is recorded.
    n_steps = max(32, min(1024, transitions // n_envs))
    model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
                n_steps=n_steps, batch_size=min(64, n_steps * n_envs), learning_rate=3e-4,
                seed=seed, device="cpu", verbose=1)
    started = time.perf_counter()
    try:
        model.learn(total_timesteps=transitions)
        model.save(output / "ppo")
        env.save(output / "vecnormalize.pkl")
        result = {"env_config": config_dict(config), "seed": seed, "requested_transitions": transitions,
                  "train_transitions": model.num_timesteps, "seconds": time.perf_counter() - started,
                  "n_envs": n_envs, "layout_split": "train"}
        save_json(output / "training.json", result)
        return result
    finally:
        env.close()
