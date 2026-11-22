import numpy as np
import pytest
import torch

from dreamnav.config import EnvConfig, MPCConfig
from dreamnav.data import EpisodeWindows, training_statistics
from dreamnav.model import GRUWorldModel, prediction_difference
from dreamnav.planner import CEMPlanner
from dreamnav.train import batch_loss


def episode():
    n = 12
    obs = np.zeros((n, 44), dtype=np.float32)
    obs[:6] = 1
    obs[6:] = 10  # Deliberate discontinuity at a waypoint switch.
    return {"observations": obs, "actions": np.zeros((n, 4), dtype=np.float32),
            "next_observations": obs + .01, "collisions": np.zeros(n, dtype=bool),
            "goal_changed": np.array([False] * 5 + [True] + [False] * 6),
            "terminated": np.array([False] * 11 + [True]), "truncated": np.zeros(n, dtype=bool),
            "sensor_validity": np.ones((n, 32), dtype=bool)}


def test_windows_never_cross_goal_or_episode_boundaries():
    data = EpisodeWindows([episode()], length=8)
    for window in data:
        valid = window["observations"][window["mask"].bool()]
        assert torch.unique(valid[:, 0]).numel() == 1
        assert 0 < window["mask"].sum() <= 6


def test_padding_does_not_affect_teacher_forced_loss():
    torch.manual_seed(2)
    model = GRUWorldModel()
    sample = {k: v[None] for k, v in EpisodeWindows([episode()], 8)[0].items()}
    original = batch_loss(model, sample, torch.tensor(1.))
    changed = {k: v.clone() for k, v in sample.items()}
    changed["next_observations"][:, 6:] = 1e6
    changed["collisions"][:, 6:] = 1
    torch.testing.assert_close(original, batch_loss(model, changed, torch.tensor(1.)))


def test_angles_wrap_across_pi():
    pred, target = torch.zeros(44), torch.zeros(44)
    pred[37], target[37] = -np.pi + .01, np.pi - .01
    assert abs(float(prediction_difference(pred, target)[37]) - .02) < 1e-5


def test_planning_does_not_contaminate_real_history():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = GRUWorldModel()
    planner = CEMPlanner(model, EnvConfig(), MPCConfig(horizon=3, candidates=8, elites=2, iterations=2))
    obs = np.zeros(44, dtype=np.float32)
    obs[:32] = 2
    obs[41] = 3
    planner.observe_transition(obs, np.zeros(4))
    before = planner.hidden.clone()
    action = planner.act(obs)
    torch.testing.assert_close(before, planner.hidden)
    assert action.shape == (4,) and np.isfinite(action).all() and abs(action).max() <= 1
    planner.observe_transition(obs, action, goal_changed=True)
    assert planner.hidden is None


def test_invalid_forecasts_use_fallback():
    model = GRUWorldModel()
    with torch.no_grad():
        model.delta.bias.fill_(float("nan"))
    planner = CEMPlanner(model, EnvConfig(), MPCConfig(horizon=2, candidates=4, elites=2, iterations=1))
    action = planner.act(np.ones(44, dtype=np.float32))
    np.testing.assert_array_equal(action, np.zeros(4))
    assert planner.fallbacks == 1 and planner.last_predictions is None


def test_train_statistics_do_not_require_validation_data():
    mean, std = training_statistics([episode()])
    assert mean.shape == std.shape == (44,)
    assert np.isfinite(mean).all() and (std >= .1).all()


def test_forecasts_use_matched_windows_and_count_short_collision_episodes(tmp_path):
    import json
    from dataclasses import asdict
    from dreamnav.evaluate import evaluate_world

    directory = tmp_path / "data/test"
    directory.mkdir(parents=True)
    # Short episodes must still count toward collision metrics.
    for index, length in enumerate([45, 7]):
        obs = np.ones((length, 44), dtype=np.float32)
        collision = np.zeros(length, dtype=bool)
        collision[-1] = index == 1
        ended = np.zeros(length, dtype=bool)
        ended[-1] = True
        np.savez(directory / f"episode-{index:06d}.npz", observations=obs, next_observations=obs,
                 actions=np.zeros((length, 4), dtype=np.float32), collisions=collision,
                 goal_changed=np.zeros(length, dtype=bool), terminated=ended,
                 truncated=np.zeros(length, dtype=bool), sensor_validity=np.ones((length, 32), dtype=bool),
                 layout_id=2_000_000 + index)
    cfg = asdict(EnvConfig())
    (directory / "manifest.json").write_text(json.dumps({"env_config": cfg}))
    checkpoint = tmp_path / "world.pt"
    torch.save({"model": GRUWorldModel().state_dict(), "env_config": cfg,
                "train_layout_ids": [0], "val_layout_ids": [1_000_000]}, checkpoint)
    output = tmp_path / "results"
    evaluate_world(checkpoint, tmp_path / "data", output, max_windows=100)
    metrics = json.loads((output / "forecast.json").read_text())
    assert {v["samples"] for v in metrics["horizons"].values()} == {2}
    assert metrics["one_step_collision_all_transitions"]["transitions"] == 52
    assert metrics["one_step_collision_all_transitions"]["positive_transitions"] == 1
