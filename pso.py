import numpy as np
from particle import Particle

NUMBER_OF_ITERACTIONS = 100
NUMBER_OF_PARTICLES = 30

# Plot range = (0, 5)
def f(x, y):
    return (x - 3.14)**2 + (y - 2.72)**2 + np.sin(3*x + 1.41) + np.sin(4*y - 1.73)

# Plot range = (-512, 512)
def f1(x1,x2):
    a=np.sqrt(np.fabs(x2+x1/2+47))
    b=np.sqrt(np.fabs(x1-(x2+47)))
    c=-(x2+47)*np.sin(a)-x1*np.sin(b)
    return c

class PSO:
    def __init__(self, function, func_range, n_particles=NUMBER_OF_PARTICLES, n_iter=NUMBER_OF_ITERACTIONS, inertia_w=0.2, cognitive_w=0.6, social_w=0.8, x_v=0.5, y_v=0.5):
        self.function = function
        self.function_range = func_range

        self.n_iter = n_iter

        self.particles = []
        self.n_particles = n_particles
        for _ in range(n_particles):
            self.particles.append(Particle(self, np.random.uniform(*func_range), np.random.uniform(*func_range), x_v, y_v, inertia_w, cognitive_w, social_w))

    def getFunction(self):
        return self.function

    def getGlobalBestX(self):
        best_fitness = 10000000
        for particle in self.particles:
            if particle.getParticleBestFitness() < best_fitness:
                best_fitness = particle.getParticleBestFitness()
                best_particle = particle
        
        return best_particle.getParticleBestX()

    def getFunctionRange(self):
        return self.function_range

    def run(self):
        for _ in range(self.n_iter):
            for particle in self.particles:
                particle.update()

    def getAllParticlesHistory(self):
        history = []
        for particle in self.particles:
            history.append(particle.getParticleHistory())
        
        return history