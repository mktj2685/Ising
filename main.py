import numpy as np
import matplotlib.pyplot as plt
from samplers.metropolis import Metropolis
from samplers.swedsen_wang import SwendsenWang
from samplers.wolff import Wolff
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

if __name__  == '__main__':
    # config
    Nx = 64
    Ny = 64
    J = 1.0
    T = 2.0
    Step = 1000

    # Create sampler
    # model = Metropolis(Nx, Ny, J, T)
    # model = SwendsenWang(Nx, Ny, J, T)
    model = Wolff(Nx, Ny, J, T)

    try:
        plt.ion()
        fig = plt.figure()
        for i in range(Step):
            model.mcstep()
            plt.imshow(model.spins)
            plt.pause(0.001)
            plt.show()

    except KeyboardInterrupt:
        plt.ioff()
        plt.close(fig)