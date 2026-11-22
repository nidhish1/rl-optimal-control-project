"""Gymnasium environment. Only the 44D observation is available to agents.

Channels: ranges[0:32], world velocity[32:35], Euler angles[35:38],
world angular velocity[38:41], world goal displacement[41:44].
Actions: normalized heading-frame vx, vy, world vz, Euler yaw rate.
"""
import contextlib
import io

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel

from .config import EnvConfig


def heading_rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def lidar_directions():
    # Four elevation rings, eight azimuths each; directions fixed to the body.
    return np.array([[np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]
                     for e in np.deg2rad([-45, -15, 15, 45])
                     for a in np.arange(8) * np.pi / 4])


def segment_distance(point, a, b):
    delta = b - a
    t = np.clip(np.dot(point - a, delta) / max(np.dot(delta, delta), 1e-9), 0, 1)
    return np.linalg.norm(point - (a + t * delta))


class DreamNavEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config=None, render_mode=None):
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.action_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
        lo = np.full(44, -np.inf, dtype=np.float32)
        hi = np.full(44, np.inf, dtype=np.float32)
        lo[:32], hi[:32] = 0, self.config.lidar_range
        lo[35:38], hi[35:38] = -np.pi, np.pi
        self.observation_space = spaces.Box(lo, hi, dtype=np.float32)
        self.aviary = None
        self.rays = lidar_directions()
        self.candidate_paths = []
        self.done = True

    @property
    def client(self):
        return self.aviary.getPyBulletClient()

    @property
    def drone(self):
        return int(self.aviary.getDroneIds()[0])

    def _state(self):
        return self.aviary._getDroneStateVector(0).copy()

    def _box(self, center, half, color):
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.client)
        # PyBullet may auto-render the collision shape when visual=-1. An explicit
        # negligible visual keeps the full collision geometry out of the camera.
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=1e-5, rgbaColor=[0, 0, 0, 0], physicsClientId=self.client) if color is None else p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=self.client)
        return p.createMultiBody(0, collision, visual, basePosition=center, physicsClientId=self.client)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.layout_id = int(options.get("layout_seed", seed if seed is not None else self.np_random.integers(1_000_000)))
        rng = np.random.default_rng(self.layout_id)
        self.start = np.array([-3.35, 0, 1.2], dtype=float)
        n = self.config.waypoints
        self.goals = np.column_stack([np.linspace(-1.1 if n > 1 else 3.2, 3.2, n),
                                     np.array([0.9 * (-1) ** i for i in range(n)]),
                                     np.array([1.2 + 0.25 * (i % 2) for i in range(n)])])
        self.goals[-1, 1] = 0
        if self.config.scenario != "empty":
            self.goals[:, 1] += rng.uniform(-0.2, 0.2, n)
        if self.aviary is None:
            with contextlib.redirect_stdout(io.StringIO()):
                self.aviary = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1,
                                        initial_xyzs=self.start[None], pyb_freq=self.config.physics_hz,
                                        ctrl_freq=self.config.control_hz, gui=self.render_mode == "human",
                                        user_debug_gui=False)
                self.pid = DSLPIDControl(drone_model=DroneModel.CF2X)
        else:
            self.aviary.reset(seed=seed)
        self.pid.reset()
        self.goal_index = self.steps = 0
        self.previous_action = np.zeros(4, dtype=np.float32)
        self.yaw_target = 0.0
        self.done = False
        self.path_length = 0.0
        self.trajectory = [self.start.copy()]
        self.candidate_paths = []
        self.obstacles = []
        self._debug_lines = []
        # A closed 8 x 6 x 3 m room, with the simulator's ground plane as floor.
        for center, half in [([0, -3, 1.5], [4, .06, 1.5]), ([0, 3, 1.5], [4, .06, 1.5]),
                             ([-4, 0, 1.5], [.06, 3, 1.5]), ([4, 0, 1.5], [.06, 3, 1.5]),
                             ([0, 0, 3], [4, 3, .06])]:
            # Collision-only walls/ceiling keep the overview camera unobstructed.
            self._box(center, half, None)
        self._box([0, 0, -.035], [4, 3, .03], [.78, .84, .89, 1])
        route = np.vstack([self.start, self.goals])
        count = {"empty": 0, "sparse": 6, "dense": 12}[self.config.scenario]
        attempts = 0
        while len(self.obstacles) < count and attempts < 1000:
            attempts += 1
            half = rng.uniform([.15, .2, .35], [.4, .5, 1.1])
            center = np.array([rng.uniform(-2.7, 2.7), rng.uniform(-2.3, 2.3), half[2]])
            # Reserve a tube along a known feasible route, including altitude changes.
            if min(segment_distance(center, a, b) for a, b in zip(route[:-1], route[1:])) < np.linalg.norm(half) + .25:
                continue
            self._box(center, half, [.24, .34, .44, 1])
            self.obstacles.append((center.copy(), half.copy()))
        for goal in self.goals:
            visual = p.createVisualShape(p.GEOM_SPHERE, radius=.10, rgbaColor=[.15, .85, .5, .8], physicsClientId=self.client)
            p.createMultiBody(0, -1, visual, basePosition=goal, physicsClientId=self.client)
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(8, 35, -40, [0, 0, .6], physicsClientId=self.client)
        obs = self._observation()
        return obs, self._info(False, False)

    def _observation(self, goal=None):
        state = self._state()
        rotation = np.asarray(p.getMatrixFromQuaternion(state[3:7])).reshape(3, 3)
        directions = self.rays @ rotation.T
        # Start rays just outside the drone's collision envelope to avoid self-hits.
        origin = state[:3] + directions * .09
        end = state[:3] + directions * self.config.lidar_range
        hits = p.rayTestBatch(origin.tolist(), end.tolist(), physicsClientId=self.client)
        ranges = np.array([self.config.lidar_range if h[0] < 0 else .09 + h[2] * (self.config.lidar_range - .09) for h in hits])
        goal = self.goals[min(self.goal_index, len(self.goals) - 1)] if goal is None else goal
        return np.concatenate([ranges, state[10:13], state[7:10], state[13:16], goal - state[:3]]).astype(np.float32)

    def _info(self, collision, success):
        return {"collision": bool(collision), "is_success": bool(success), "waypoints_reached": self.goal_index,
                "layout_id": self.layout_id, "position": self._state()[:3].copy(),
                "path_length": self.path_length, "elapsed_seconds": self.steps / self.config.navigation_hz,
                "sensor_validity": np.ones(32, dtype=bool)}

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset() before stepping a finished episode.")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("Action must contain four finite numbers.")
        action = np.clip(action, -1, 1)
        cfg = self.config
        old_goal = self.goals[self.goal_index].copy()
        old_position = self._state()[:3]
        distance_before = np.linalg.norm(old_goal - old_position)
        collision = False
        for _ in range(cfg.control_hz // cfg.navigation_hz):
            state = self._state()
            velocity = heading_rotation(state[9]) @ (action[:3] * [cfg.max_speed_xy, cfg.max_speed_xy, cfg.max_speed_z])
            self.yaw_target += float(action[3]) * cfg.max_yaw_rate / cfg.control_hz
            self.yaw_target = (self.yaw_target + np.pi) % (2 * np.pi) - np.pi
            rpm, _, _ = self.pid.computeControl(1 / cfg.control_hz, state[:3], state[3:7], state[10:13], state[13:16],
                                                target_pos=state[:3], target_vel=velocity,
                                                target_rpy=np.array([0, 0, self.yaw_target]),
                                                target_rpy_rates=np.array([0, 0, action[3] * cfg.max_yaw_rate]))
            self.aviary.step(rpm[None])
            pos = self._state()[:3]
            contact = bool(p.getContactPoints(bodyA=self.drone, physicsClientId=self.client))
            collision = contact or bool(abs(pos[0]) > 3.96 or abs(pos[1]) > 2.96 or pos[2] < .08 or pos[2] > 2.96)
            if collision:
                break
        self.steps += 1
        model_next = self._observation(goal=old_goal)
        position = self._state()[:3]
        self.path_length += float(np.linalg.norm(position - old_position))
        self.trajectory.append(position.copy())
        distance_after = np.linalg.norm(model_next[41:44])
        progress = float(distance_before - distance_after)
        reached = bool(distance_after < cfg.goal_radius and not collision)
        if reached:
            self.goal_index += 1
        success = self.goal_index == len(self.goals)
        reward = (cfg.progress_weight * progress - cfg.collision_weight * collision
                  - cfg.smoothness_weight * float(np.square(action - self.previous_action).sum())
                  - cfg.clearance_weight * max(0., cfg.clearance_distance - float(model_next[:32].min()))
                  - cfg.time_cost + cfg.goal_bonus * reached)
        self.previous_action = action.copy()
        terminated = bool(collision or success)
        truncated = bool(self.steps >= cfg.max_steps and not terminated)
        self.done = terminated or truncated
        info = self._info(collision, success)
        # Keep the prediction target tied to the OLD goal. A goal switch is exogenous.
        info.update(model_next_observation=model_next, goal_changed=reached and not success, goal_progress=progress)
        return self._observation(), float(reward), terminated, truncated, info

    def set_candidate_paths(self, paths):
        self.candidate_paths = np.asarray(paths)
        if self.render_mode == "human":
            for line in self._debug_lines:
                p.removeUserDebugItem(line, physicsClientId=self.client)
            self._debug_lines = []
            for i, path in enumerate(self.candidate_paths):
                for a, b in zip(path[:-1], path[1:]):
                    self._debug_lines.append(p.addUserDebugLine(a, b, [1, .65, .1] if i == 0 else [.3, .7, 1],
                                                              lineWidth=2 if i == 0 else 1, physicsClientId=self.client))

    def render(self):
        if self.aviary is None:
            return None
        from PIL import Image, ImageDraw, ImageFont
        width, height = 960, 640
        view = p.computeViewMatrixFromYawPitchRoll([0, 0, .7], 8, 35, -48, 0, 2)
        projection = p.computeProjectionMatrixFOV(58, width / height, .1, 30)
        rgba = p.getCameraImage(width, height, view, projection, renderer=p.ER_TINY_RENDERER,
                               physicsClientId=self.client)[2]
        image = Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3])
        draw = ImageDraw.Draw(image)
        title_font = ImageFont.load_default(size=22)
        label_font = ImageFont.load_default(size=16)
        transform = np.array(projection).reshape(4, 4, order="F") @ np.array(view).reshape(4, 4, order="F")

        def pixels(points):
            homogeneous = np.column_stack([points, np.ones(len(points))]) @ transform.T
            ndc = homogeneous[:, :2] / homogeneous[:, 3:4]
            return [((x + 1) * width / 2, (1 - y) * height / 2) for x, y in ndc]

        if len(self.trajectory) > 1:
            draw.line(pixels(self.trajectory), fill="#12b886", width=3)
        boundary = np.array([[-4, -3, .02], [4, -3, .02], [4, 3, .02], [-4, 3, .02], [-4, -3, .02]])
        draw.line(pixels(boundary), fill="#607d8b", width=2)
        for i in reversed(range(len(self.candidate_paths))):
            draw.line(pixels(self.candidate_paths[i]), fill="#f59f00" if i == 0 else "#74c0fc", width=3 if i == 0 else 1)
        for i, point in enumerate(pixels(self.goals)):
            draw.text((point[0] + 8, point[1] - 12), str(i + 1), fill="#064e3b", font=label_font)
        x, y = pixels([self._state()[:3]])[0]
        draw.ellipse([x - 7, y - 7, x + 7, y + 7], outline="#112236", width=2)
        draw.rectangle([0, 0, width, 64], fill="#112236")
        draw.text((18, 7), "DREAMNAV  |  " + getattr(self, "controller_label", "Quadcopter navigation"), fill="white", font=title_font)
        draw.text((18, 37), f"Waypoints {self.goal_index}/{len(self.goals)}   Time {self.steps / self.config.navigation_hz:.1f}s   Layout {self.layout_id}", fill="#aad9ef", font=label_font)
        draw.rectangle([0, height - 30, width, height], fill="#112236")
        legend = "Green: actual flight"
        if len(self.candidate_paths):
            legend += "   Orange: selected forecast   Blue: candidate forecasts"
        draw.text((18, height - 24), legend, fill="white", font=label_font)
        return np.asarray(image)

    def close(self):
        if self.aviary is not None:
            if p.isConnected(self.client):
                self.aviary.close()
            self.aviary = None
