import math
import random
import numpy as np


class Ising:
    
    def __init__(
        self,
        Nx: int,
        Ny: int,
        J: float,
        T: float,
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
        
        # Define neighbors
        # self.nbrs rule lattice type (e.g. triangle, kagome, ...) and boundary condition (open, periodic). 
        self.nbrs = lambda x, y: [
            (x, (y+1)%self.Ny), # top
            ((x+1)%self.Nx, y)  # right
        ]      

        # Energy of current spins configuration.
        self.e = self.energy()

    def energy(self):
        e = 0.0
        for x in range(self.Nx-1):
            for y in range(self.Ny-1):
                for x_, y_ in self.nbrs(x, y):
                    e += -1.0 * self.J * self.spins[x, y] * self.spins[x_, y_]

        return e

    def metropolis(self) -> None:
        # Select one vertex at random.
        x = random.randint(0, self.Nx-1)
        y = random.randint(0, self.Ny-1)

        # Flip spin
        self.spins[x, y] *= -1
        
        # Calculate the energy after spin flipped.
        e = self.energy()

        # Calculate the energy difference.
        de = e - self.e

        # If energy decreses, accept updated edges.
        if de < 0:
            self.e = e

        # If energy dosen't decreses, accept update edges with probabilty P = exp(-ΔE/T).
        else:
            r = random.random()

            # accept
            if r < math.exp(-de / self.T):
                self.e = e

            # reject
            else:
                self.spins[x, y] *= -1

    def mcstep(self) -> None:
        self.metropolis()

if __name__ == '__main__':
    Nx = 16
    Ny = 16
    J = 1.0
    T = 1.0

    model = Ising(Nx, Ny, J, T)

    import matplotlib.pyplot as plt
    plt.ion()
    for i in range(100000):
        model.mcstep()
        print(f'M = {model.spins.sum()}')
        # plt.imshow(model.spins)
        # plt.pause(0.001)
        # plt.show()