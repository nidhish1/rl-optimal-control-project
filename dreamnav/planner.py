"""CEM-MPC over learned observations, never simulator lookahead."""
import numpy as np
import torch

from .config import MPCConfig


class CEMPlanner:
    def __init__(self, model, env_config, config=None, seed=0):
        self.model = model.eval()
        self.env_config = env_config
        self.config = config or MPCConfig()
        self.device = next(model.parameters()).device
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.reset()

    def reset(self):
        self.hidden = None
        self.mean = torch.zeros(self.config.horizon, 4, device=self.device)
        self.previous_action = torch.zeros(4, device=self.device)
        self.last_predictions = None
        self.fallbacks = 0

    @torch.inference_mode()
    def observe_transition(self, observation, action, goal_changed=False):
        if goal_changed:
            self.hidden = None
            self.mean.zero_()
        else:
            obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)[None]
            act = torch.as_tensor(action, dtype=torch.float32, device=self.device)[None]
            _, _, self.hidden = self.model.step(obs, act, self.hidden)
        self.previous_action = torch.as_tensor(action, dtype=torch.float32, device=self.device)

    @torch.inference_mode()
    def _rollout(self, observation, actions):
        n, horizon, _ = actions.shape
        obs = observation[None].expand(n, -1)
        hidden = None if self.hidden is None else self.hidden.expand(-1, n, -1).contiguous()
        score = torch.zeros(n, device=self.device)
        valid = torch.ones(n, dtype=torch.bool, device=self.device)
        survival = torch.ones(n, device=self.device)
        previous = self.previous_action[None].expand(n, -1)
        predictions, risks = [], []
        cfg = self.env_config
        for k in range(horizon):
            prediction, logits, hidden = self.model.step(obs, actions[:, k], hidden)
            risk = torch.sigmoid(logits)
            finite = torch.isfinite(prediction).all(-1) & torch.isfinite(risk)
            # Reject implausible forecasts that could inflate the score.
            plausible = ((prediction[:, 41:44] - obs[:, 41:44]).norm(dim=-1) < .6)
            plausible &= (prediction[:, 32:35].norm(dim=-1) < 4)
            plausible &= (prediction[:, :32].min(-1).values > -.25)
            plausible &= (prediction[:, :32].max(-1).values < cfg.lidar_range + .5)
            valid &= finite & plausible
            prediction = torch.nan_to_num(prediction)
            risk = torch.nan_to_num(risk, nan=1.)
            progress = obs[:, 41:44].norm(dim=-1) - prediction[:, 41:44].norm(dim=-1)
            clearance = (cfg.clearance_distance - prediction[:, :32].min(-1).values).clamp_min(0)
            smoothness = (actions[:, k] - previous).square().sum(-1)
            reward = (cfg.progress_weight * progress - cfg.collision_weight * risk
                      - cfg.smoothness_weight * smoothness - cfg.clearance_weight * clearance - cfg.time_cost)
            score += self.config.discount ** k * survival * reward
            # Stop scoring progress after arrival or collision.
            arrived = prediction[:, 41:44].norm(dim=-1) < cfg.goal_radius
            score += self.config.discount ** k * survival * cfg.goal_bonus * arrived
            survival *= (1 - risk) * (~arrived)
            predictions.append(prediction)
            risks.append(risk)
            obs, previous = prediction, actions[:, k]
        score[~valid] = -torch.inf
        return score, torch.stack(predictions, 1), torch.stack(risks, 1)

    @torch.inference_mode()
    def act(self, observation):
        if not np.isfinite(observation).all():
            self.fallbacks += 1
            self.last_predictions = None
            return np.zeros(4, dtype=np.float32)
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        cfg = self.config
        mean = self.mean.clone()
        std = torch.full_like(mean, .6)
        best_score = -float("inf")
        best_sequence = best_prediction = best_risk = None
        for _ in range(cfg.iterations):
            noise = torch.randn((cfg.candidates, cfg.horizon, 4), generator=self.generator, device=self.device)
            actions = (mean + std * noise).clamp(-1, 1)
            actions[0] = mean.clamp(-1, 1)
            actions[1].zero_()  # Include a hover candidate.
            scores, predictions, risks = self._rollout(obs, actions)
            valid_count = int(torch.isfinite(scores).sum())
            if valid_count == 0:
                continue
            elite_ids = scores.topk(min(cfg.elites, valid_count)).indices
            elites = actions[elite_ids]
            mean = elites.mean(0)
            std = elites.std(0, unbiased=False).clamp_min(cfg.min_std)
            winner = int(elite_ids[0])
            if float(scores[winner]) > best_score:
                best_score = float(scores[winner])
                best_sequence = actions[winner].clone()
                best_prediction = predictions[elite_ids[:8]].clone()
                best_risk = float(risks[winner, 0])
        if best_sequence is None or best_risk > cfg.risk_threshold:
            self.fallbacks += 1
            self.last_predictions = None
            self.mean.zero_()
            return np.zeros(4, dtype=np.float32)
        self.last_predictions = best_prediction.cpu().numpy()
        self.mean = torch.cat([best_sequence[1:], best_sequence[-1:]], 0)
        return best_sequence[0].cpu().numpy()
