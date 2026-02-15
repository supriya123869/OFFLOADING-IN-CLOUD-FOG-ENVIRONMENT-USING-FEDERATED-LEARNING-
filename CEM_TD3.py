# ============================================================
# CEM-TD3 for Cloud-Fog Offloading
# Hyperparameter Optimization using Cross Entropy Method
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# PRIORITIZED REPLAY BUFFER
# ============================================================

class PERBuffer:
    def __init__(self, capacity=100000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def add(self, state, action, reward, next_state):
        priority = max(self.priorities, default=1.0)

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state))
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = (state, action, reward, next_state)
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        states, actions, rewards, next_states = zip(*samples)

        return (
            torch.FloatTensor(states).to(device),
            torch.FloatTensor(actions).to(device),
            torch.FloatTensor(rewards).unsqueeze(1).to(device),
            torch.FloatTensor(next_states).to(device),
        )

    def __len__(self):
        return len(self.buffer)

# ============================================================
# ACTOR & CRITIC NETWORKS (Dynamic Architecture)
# ============================================================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hp):
        super().__init__()
        layers = []
        input_dim = state_dim

        for _ in range(hp["num_layers"]):
            layers.append(nn.Linear(input_dim, hp["neurons"]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(hp["dropout"]))
            input_dim = hp["neurons"]

        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Sigmoid())  # action in [0,1]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hp):
        super().__init__()
        layers = []
        input_dim = state_dim + action_dim

        for _ in range(hp["num_layers"]):
            layers.append(nn.Linear(input_dim, hp["neurons"]))
            layers.append(nn.ReLU())
            input_dim = hp["neurons"]

        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=1))

# ============================================================
# TD3 AGENT
# ============================================================

class TD3Agent:

    def __init__(self, state_dim, action_dim, hp):

        self.hp = hp

        self.actor = Actor(state_dim, action_dim, hp).to(device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic1 = Critic(state_dim, action_dim, hp).to(device)
        self.critic2 = Critic(state_dim, action_dim, hp).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor_opt = optim.Adam(self.actor.parameters(),
                                     lr=hp["lr"],
                                     betas=(hp["beta"], 0.999))
        self.critic_opt = optim.Adam(
            list(self.critic1.parameters()) +
            list(self.critic2.parameters()),
            lr=hp["lr"],
            betas=(hp["beta"], 0.999)
        )

        self.buffer = PERBuffer(alpha=hp["per_alpha"])

        self.gamma = hp["gamma"]
        self.tau = hp["tau"]
        self.noise_std = hp["noise"]

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state).detach().cpu().numpy()[0]
        action += np.random.normal(0, self.noise_std)
        return np.clip(action, 0, 1)

    def train(self, batch_size):

        if len(self.buffer) < batch_size:
            return

        states, actions, rewards, next_states = \
            self.buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.noise_std).clamp(-0.2,0.2)
            next_actions = (self.actor_target(next_states) + noise).clamp(0,1)

            q1_target = self.critic1_target(next_states, next_actions)
            q2_target = self.critic2_target(next_states, next_actions)
            q_target = rewards + self.gamma * torch.min(q1_target, q2_target)

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        critic_loss = nn.MSELoss()(q1, q_target) + \
                      nn.MSELoss()(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Delayed Actor update
        actor_loss = -self.critic1(states, self.actor(states)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft update
        for param, target in zip(self.actor.parameters(),
                                 self.actor_target.parameters()):
            target.data.copy_(self.tau*param.data + (1-self.tau)*target.data)

        for param, target in zip(self.critic1.parameters(),
                                 self.critic1_target.parameters()):
            target.data.copy_(self.tau*param.data + (1-self.tau)*target.data)

        for param, target in zip(self.critic2.parameters(),
                                 self.critic2_target.parameters()):
            target.data.copy_(self.tau*param.data + (1-self.tau)*target.data)

# ============================================================
# CROSS ENTROPY METHOD FOR HYPERPARAMETER SEARCH
# ============================================================

class CEMOptimizer:

    def __init__(self, population=20, elite_frac=0.2):
        self.population = population
        self.elite_frac = elite_frac

    def sample_hp(self):
        return {
            "lr": 10**np.random.uniform(-5, -3),
            "dropout": np.random.uniform(0.0, 0.5),
            "num_layers": np.random.randint(3, 8),
            "neurons": np.random.randint(64, 513),
            "beta": np.random.uniform(0.7, 0.999),
            "tau": np.random.uniform(0.001, 0.01),
            "batch_size": np.random.randint(32, 257),
            "gamma": np.random.uniform(0.90, 0.99),
            "noise": np.random.uniform(0.1, 0.5),
            "per_alpha": np.random.uniform(0, 1)
        }

    def optimize(self, env, generations=5):

        best_hp = None
        best_score = -np.inf

        for gen in range(generations):

            population = [self.sample_hp()
                          for _ in range(self.population)]

            scores = []

            for hp in population:

                agent = TD3Agent(5, 1, hp)

                total_reward = 0
                state = env.reset()

                for _ in range(300):
                    action = agent.select_action(state)
                    next_state, reward = env.step(action)
                    agent.buffer.add(state, action, reward, next_state)
                    agent.train(hp["batch_size"])
                    state = next_state
                    total_reward += reward

                scores.append(total_reward)

            elite_idx = np.argsort(scores)[-int(self.population*self.elite_frac):]

            elite_scores = [scores[i] for i in elite_idx]
            elite_hp = [population[i] for i in elite_idx]

            gen_best = max(elite_scores)

            if gen_best > best_score:
                best_score = gen_best
                best_hp = elite_hp[np.argmax(elite_scores)]

            print(f"Generation {gen+1} Best Reward: {gen_best}")

        return best_hp