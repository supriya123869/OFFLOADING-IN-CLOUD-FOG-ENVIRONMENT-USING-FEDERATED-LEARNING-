import numpy as np

class PERBuffer:
    def __init__(self, max_size, alpha):
        self.max_size = max_size
        self.alpha = alpha
        self.ptr = 0
        self.size = 0

        self.storage = []
        self.priorities = np.zeros(max_size)

    def add(self, data):
        max_prio = self.priorities.max() if self.size else 1.0

        if self.size < self.max_size:
            self.storage.append(data)
        else:
            self.storage[self.ptr] = data

        self.priorities[self.ptr] = max_prio
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self.size, batch_size, p=probs)

        batch = [self.storage[i] for i in indices]
        return batch, indices

    def update(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self.priorities[i] = p