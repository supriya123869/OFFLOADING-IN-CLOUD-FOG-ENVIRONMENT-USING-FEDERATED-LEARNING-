import torch
import copy
import torch.nn.functional as F
from agents.networks import Actor, Critic
from agents.per_buffer import PERBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3:
    def __init__(self, s_dim, a_dim, max_action, hp):

        self.actor = Actor(s_dim, a_dim, max_action,
                           hp["layers"], hp["neurons"], hp["dropout"]).to(device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(s_dim, a_dim,
                             hp["layers"], hp["neurons"], hp["dropout"]).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(),
            lr=hp["lr"],
            betas=(hp["beta"], 0.999)
        )

        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(),
            lr=hp["lr"],
            betas=(hp["beta"], 0.999)
        )

        self.memory = PERBuffer(1_000_000, hp["per_alpha"])

        self.gamma = hp["gamma"]
        self.tau = hp["tau"]
        self.noise = hp["noise"]
        self.batch = hp["batch"]
        self.max_action = max_action
        self.total_it = 0