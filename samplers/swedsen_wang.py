import math
import random
from typing import List, Tuple
import numpy as np
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))
from utils.unionfind import UnionFind


class SwendsenWang:
    
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

        # Define neighbors
        # self.nbrs rule lattice type (e.g. triangle, kagome, ...) and boundary condition (open, periodic). 
        self.nbrs = lambda x, y: [
            (x, (y+1)%self.Ny), # top
            ((x+1)%self.Nx, y)  # right
        ]

    def swendsen_wang(self) -> None:
        # Create Union-Find
        uf = UnionFind(self.Nx*self.Ny)

        # Calculate Padd.
        Padd = 1.0 - math.exp(-2.0 * self.J / self.T)

        # Loop run vertices belongs bulk.
        for x in range(self.Nx):
            for y in range(self.Ny):
                for x_, y_ in self.nbrs(x, y):
                    if self.spins[x, y] == self.spins[x_, y_] and random.random() < Padd:
                        i = y * self.Nx + x
                        j = y_ * self.Nx + x_
                        uf.union(i, j)

        # Cluster spins 1 with probability 1/2 otherwise -1.
        for root in uf.roots():
            s = 1 if random.random() > 0.5 else -1
            for idx in uf.members(root):
                x = idx % self.Nx
                y = idx // self.Nx
                self.spins[x, y] = s

    def mcstep(self) -> None:
        self.swendsen_wang()

if __name__ == '__main__':
    Nx = 16
    Ny = 16
    J = 1.0
    T = 1.0
    model = SwendsenWang(Nx, Ny, J, T)

    import matplotlib.pyplot as plt
    plt.ion()
    for i in range(100):
        model.mcstep()
        print(f'M = {model.spins.sum()}')
        # plt.imshow(model.spins)
        # plt.pause(0.001)
        # plt.show()