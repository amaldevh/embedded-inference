# my_project/policy.py
import torch

class Policy(torch.nn.Module):
    def __init__(self, state_dim=13, action_dim=4):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim),
        )

    def forward(self, state):
        return self.net(state)

def create_policy(state_dim=13, action_dim=4):
    return Policy(state_dim, action_dim)
