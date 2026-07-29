"""Small policy network used to smoke-test export and deployment."""

import torch


class ExamplePolicy(torch.nn.Module):
    def __init__(self, state_dim=13, hidden_dim=128, action_dim=4):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        return self.network(state)


def create_model(state_dim=13, hidden_dim=128, action_dim=4):
    return ExamplePolicy(state_dim, hidden_dim, action_dim)
