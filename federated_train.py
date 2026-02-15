import numpy as np
from clustering.spectral import SpectralAgentSelector
from federation.aggregator import federated_average


def federated_training(agents):

    features = []

    for agent in agents:
        flat = []
        for p in agent.actor.parameters():
            flat.append(p.detach().cpu().numpy().flatten())
        features.append(np.concatenate(flat))

    features = np.array(features)

    selector = SpectralAgentSelector(k=2)
    selected_ids = selector.select(features)

    selected_agents = [agents[i] for i in selected_ids]

    global_weights = federated_average(selected_agents)

    return global_weights