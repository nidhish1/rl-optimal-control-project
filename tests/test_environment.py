import numpy as np
import pybullet as p
import pytest
from stable_baselines3.common.env_checker import check_env

from dreamnav.config import EnvConfig, layout_seed
from dreamnav.data import guided_action
from dreamnav.env import DreamNavEnv


@pytest.fixture
def env():
    instance = DreamNavEnv(EnvConfig(scenario="empty", max_steps=300))
    yield instance
    instance.close()


def test_gym_contract_and_seed_reproducibility(env):
    check_env(env, warn=True)
    first, _ = env.reset(seed=17)
    first_step = env.step(np.array([.4, -.2, .1, .2], dtype=np.float32))[0]
    second, _ = env.reset(seed=17)
    second_step = env.step(np.array([.4, -.2, .1, .2], dtype=np.float32))[0]
    np.testing.assert_allclose(first, second, atol=1e-6)
    np.testing.assert_allclose(first_step, second_step, atol=1e-6)


def test_hover_and_velocity_control(env):
    _, info = env.reset(seed=0)
    initial = info["position"].copy()
    for _ in range(30):
        obs, _, terminated, truncated, info = env.step(np.zeros(4))
        assert not terminated and not truncated
    assert np.linalg.norm(info["position"] - initial) < .05
    for _ in range(20):
        obs, _, terminated, _, info = env.step(np.array([.5, 0, 0, 0]))
    assert info["position"][0] > initial[0] + .3
    assert abs(info["position"][2] - initial[2]) < .15


def test_yaw_action_is_not_speed(env):
    obs, info = env.reset(seed=0)
    initial = info["position"].copy()
    for _ in range(20):
        obs, _, terminated, _, info = env.step(np.array([0, 0, 0, .5]))
        assert not terminated
    assert obs[37] > .2
    assert np.linalg.norm(info["position"] - initial) < .1


def test_waypoint_switch_target_and_mission_completion(env):
    obs, _ = env.reset(seed=3)
    switches = 0
    for _ in range(300):
        old_goal = env.goals[env.goal_index].copy()
        obs, _, terminated, truncated, info = env.step(guided_action(obs, env.config))
        if info["goal_changed"]:
            switches += 1
            np.testing.assert_allclose(info["model_next_observation"][41:44], old_goal - info["position"], atol=1e-5)
            assert np.linalg.norm(obs[41:44] - info["model_next_observation"][41:44]) > .5
        if terminated or truncated:
            break
    assert info["is_success"] and not info["collision"]
    assert switches == 2 and info["waypoints_reached"] == 3
    with pytest.raises(RuntimeError):
        env.step(np.zeros(4))


def test_collision_timeout_and_invalid_action(env):
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.array([np.nan, 0, 0, 0]))
    env._box(env.start, [.2, .2, .2], [1, 0, 0, 1])
    _, _, terminated, truncated, info = env.step(np.zeros(4))
    assert terminated and info["collision"] and not truncated
    env.config.max_steps = 1
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step(np.zeros(4))
    assert truncated and not terminated and not info["collision"]


def test_lidar_detects_inserted_obstacle(env):
    obs, _ = env.reset(seed=0)
    # Ray 16 points forward and up by 15 degrees.
    ray = env.rays[16]
    env._box(env.start + ray, [.12, .12, .12], [1, 0, 0, 1])
    after = env._observation()
    assert .7 < after[16] < 1.1
    assert after[16] < obs[16]


def test_split_ids_do_not_overlap():
    assert layout_seed("train", 999999) < layout_seed("val", 0)
    assert layout_seed("val", 999999) < layout_seed("test", 0)
    with pytest.raises(ValueError):
        layout_seed("train", 1_000_000)


def test_cleanup_after_simulator_disconnect(env):
    env.reset(seed=0)
    p.disconnect(physicsClientId=env.client)
    env.close()
    env.close()
    assert env.aviary is None


def test_multiple_clients_are_independent():
    a = DreamNavEnv(EnvConfig(scenario="empty"))
    b = DreamNavEnv(EnvConfig(scenario="empty"))
    try:
        a.reset(seed=0)
        b.reset(seed=1)
        initial_b = b._state().copy()
        for _ in range(10):
            a.step(np.array([.5, 0, 0, 0]))
        np.testing.assert_array_equal(initial_b, b._state())
    finally:
        a.close()
        b.close()
