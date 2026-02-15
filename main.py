from utils.seed import set_seed
from cem.cem_optimizer import CEM

def main():

    set_seed(42)

    cem = CEM()
    population = cem.sample()

    print("Sampled Hyperparameters:")
    for hp in population:
        print(hp)


if __name__ == "__main__":
    main()