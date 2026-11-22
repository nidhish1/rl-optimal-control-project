from dataclasses import asdict, dataclass
import json
from pathlib import Path

SPLIT_OFFSET = {"train": 0, "val": 1_000_000, "test": 2_000_000}


def layout_seed(split: str, index: int) -> int:
    if split not in SPLIT_OFFSET or not 0 <= index < 1_000_000:
        raise ValueError("Use train/val/test and a layout index in [0, 1000000).")
    return SPLIT_OFFSET[split] + index


@dataclass
class EnvConfig:
    physics_hz: int = 240
    control_hz: int = 60
    navigation_hz: int = 10
    max_steps: int = 600
    waypoints: int = 3
    scenario: str = "sparse"
    max_speed_xy: float = 0.8
    max_speed_z: float = 0.4
    max_yaw_rate: float = 0.6
    lidar_range: float = 5.0
    goal_radius: float = 0.3
    progress_weight: float = 5.0
    collision_weight: float = 20.0
    smoothness_weight: float = 0.03
    clearance_weight: float = 0.1
    clearance_distance: float = 0.3
    time_cost: float = 0.01
    goal_bonus: float = 5.0

    def __post_init__(self):
        if min(self.physics_hz, self.control_hz, self.navigation_hz) <= 0:
            raise ValueError("Frequencies must be positive.")
        if self.physics_hz % self.control_hz or self.control_hz % self.navigation_hz:
            raise ValueError("physics_hz must divide by control_hz, which must divide by navigation_hz.")
        if self.scenario not in {"empty", "sparse", "dense"}:
            raise ValueError("Unknown scenario.")
        if not 1 <= self.waypoints <= 8 or self.max_steps < 1:
            raise ValueError("Use 1-8 waypoints and a positive episode length.")


@dataclass
class MPCConfig:
    horizon: int = 10
    candidates: int = 128
    elites: int = 16
    iterations: int = 4
    discount: float = 0.99
    min_std: float = 0.08
    risk_threshold: float = 0.85

    def __post_init__(self):
        if self.horizon < 1 or self.iterations < 1 or not 2 <= self.elites <= self.candidates:
            raise ValueError("MPC needs positive horizon/iterations and 2 <= elites <= candidates.")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def config_dict(config):
    return asdict(config)
