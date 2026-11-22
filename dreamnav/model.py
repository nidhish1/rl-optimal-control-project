import numpy as np
import torch
from torch import nn


class GRUWorldModel(nn.Module):
    def __init__(self, mean=None, std=None):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(np.zeros(44) if mean is None else mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(np.ones(44) if std is None else std, dtype=torch.float32))
        self.encoder = nn.Sequential(nn.Linear(44, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU())
        self.gru = nn.GRU(68, 128, batch_first=True)
        self.delta = nn.Linear(128, 44)
        self.collision = nn.Linear(128, 1)
        nn.init.normal_(self.delta.weight, std=.001)
        nn.init.zeros_(self.delta.bias)
        nn.init.constant_(self.collision.bias, -3.)

    def forward(self, observations, actions, hidden=None):
        encoded = self.encoder((observations - self.mean) / self.std)
        features, hidden = self.gru(torch.cat([encoded, actions], dim=-1), hidden)
        prediction = observations + self.delta(features) * self.std
        # Keep Euler angles in [-pi, pi].
        angles = torch.atan2(torch.sin(prediction[..., 35:38]), torch.cos(prediction[..., 35:38]))
        prediction = torch.cat([prediction[..., :35], angles, prediction[..., 38:]], dim=-1)
        return prediction, self.collision(features).squeeze(-1), hidden

    def step(self, observation, action, hidden=None):
        prediction, collision, hidden = self(observation[:, None], action[:, None], hidden)
        return prediction[:, 0], collision[:, 0], hidden


def prediction_difference(prediction, target):
    error = prediction - target
    angle = torch.atan2(torch.sin(error[..., 35:38]), torch.cos(error[..., 35:38]))
    return torch.cat([error[..., :35], angle, error[..., 38:]], dim=-1)


def load_world_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = GRUWorldModel().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint
