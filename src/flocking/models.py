import torch
from torch import nn
import pfrl

STATE_DIM = 11
ACTION_DIM = 9


class QFunction(nn.Module):
    """
    4-layer MLP Q-network for DDQN.

    Architecture: 11 → 64 → 32 → 16 → 9
    Each hidden layer uses BatchNorm + LeakyReLU.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Linear(16, ACTION_DIM),
        )

    def forward(self, x):
        return pfrl.action_value.DiscreteActionValue(self.net(x))


class QFunction_LSTM(nn.Module):
    """
    LSTM-based Q-network variant.
    Kept for experimentation — standard QFunction is used in the default training run.
    """

    def __init__(self):
        super().__init__()
        self.linear_in = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
        )
        self.lstm = pfrl.nn.RecurrentSequential(
            nn.LSTM(input_size=64, hidden_size=64, num_layers=3, batch_first=True),
        )
        self.linear_out = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, ACTION_DIM),
        )

    def forward(self, x, recurrent_state):
        h = self.linear_in(x)
        h, new_state = self.lstm(h, recurrent_state)
        return pfrl.action_value.DiscreteActionValue(self.linear_out(h)), new_state
