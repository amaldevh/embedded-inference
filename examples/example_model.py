"""Small policy network used to smoke-test export and deployment."""

from __future__ import annotations

import torch


class ExamplePolicy(torch.nn.Module):
    def __init__(
        self, state_dim: int = 13, hidden_dim: int = 128, action_dim: int = 4
    ) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


def create_model(
    state_dim: int = 13, hidden_dim: int = 128, action_dim: int = 4
) -> ExamplePolicy:
    return ExamplePolicy(state_dim, hidden_dim, action_dim)
