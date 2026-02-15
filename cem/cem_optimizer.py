import numpy as np

class CEM:
    def __init__(self, pop=10, elite_frac=0.2):
        self.pop = pop
        self.elite = int(pop * elite_frac)

    def sample(self):
        pop = []
        for _ in range(self.pop):
            hp = {
                "lr": 10 ** np.random.uniform(-5, -3),
                "dropout": np.random.uniform(0, 0.5),
                "layers": np.random.randint(3, 8),
                "neurons": np.random.randint(64, 513),
                "beta": np.random.uniform(0.7, 0.999),
                "tau": np.random.uniform(0.001, 0.01),
                "batch": np.random.randint(32, 257),
                "gamma": np.random.uniform(0.90, 0.99),
                "noise": np.random.uniform(0.1, 0.5),
                "per_alpha": np.random.uniform(0, 1)
            }
            pop.append(hp)
        return pop