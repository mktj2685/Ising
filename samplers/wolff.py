import math
import random
from typing import List, Tuple
import numpy as np


class Wolff:
    
    def __init__(
        self,
        Nx: int,
        Ny: int,
        J: float,
        T: float
    ) -> None:
        # Set lattice shape.
        self.Nx = Nx
        self.Ny = Ny

        # Set coupling constant.
        #   J > 0 : ferromagnetic
        #   J < 0 : antiferromagnetic
        self.J = J

        # Set temperature
        self.T = T

        # Define Ising spings.
        self.spins = 2 * np.random.randint(2, size=(Nx, Ny), dtype=np.int8) - 1

        # Define lattice type (e.g. square, triangle, ...) and boundary condition (open, periodic, ...)
        self.nbrs = lambda x, y: [
                ((x-1)%self.Nx, y),     # left
                (x, (y+1)%self.Ny),     # top
                ((x+1)%self.Nx, y),     # right
                ((x, (y-1)%self.Ny))    # bottom
            ]

    def wolff(self) -> None:
        # Select one vertex at random.
        x = random.randint(0, self.Nx-1)
        y = random.randint(0, self.Ny-1)

        # Add to seed.
        seed = []
        cluster = []
        seed.append((x, y))
        cluster.append((x, y))

        # Calculate Padd.
        Padd = 1.0 - math.exp(-2.0 * self.J / self.T)

        # Loop run until seed is empty.
        while seed:
            # Choice one vertex from seed.
            idx = random.randint(0, len(seed)-1)
            coord = seed.pop(idx)

            # Loop run nearest-neighbor spins.
            for nbr in self.nbrs(*coord):
                # If they are pointing in the same direction as the seed spin,
                # add them to the cluster with probability Padd = 1 - exp(-2βJ/T)
                if self.spins[coord] == self.spins[nbr] and nbr not in cluster and random.random() < Padd:
                    seed.append(nbr)
                    cluster.append(nbr)

        # Aligne cluster 1 with probability 1/2 otherwise -1.
        s = 1 if random.random() < 0.5 else -1
        for coord in cluster:
            self.spins[coord] = s

    def mcstep(self) -> None:
        self.wolff()

if __name__ == '__main__':
    Nx = 16
    Ny = 16
    J = 1.0
    T = 1.0
    model = Wolff(Nx, Ny, J, T)

    import matplotlib.pyplot as plt
    plt.ion()
    for i in range(1000):
        model.mcstep()
        print(f'M = {model.spins.sum()}')
        # plt.imshow(model.spins)
        # plt.pause(0.001)
        # plt.show()