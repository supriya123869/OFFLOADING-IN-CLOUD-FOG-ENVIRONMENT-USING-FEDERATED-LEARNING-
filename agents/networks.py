import torch
import torch.nn as nn


def build_mlp(input_dim, output_dim, layers, neurons, dropout):
    net = []
    last = input_dim

    for _ in range(layers):
        net.append(nn.Linear(last, neurons))
        net.append(nn.ReLU())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
        last = neurons

    net.append(nn.Linear(last, output_dim))
    return nn.Sequential(*net)


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, max_action,
                 layers, neurons, dropout):
        super().__init__()
        self.net = build_mlp(s_dim, a_dim,
                             layers, neurons, dropout)
        self.max_action = max_action

    def forward(self, s):
        return self.max_action * torch.tanh(self.net(s))


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim,
                 layers, neurons, dropout):
        super().__init__()
        self.q1 = build_mlp(s_dim + a_dim, 1,
                            layers, neurons, dropout)
        self.q2 = build_mlp(s_dim + a_dim, 1,
                            layers, neurons, dropout)

    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        return self.q1(sa), self.q2(sa)