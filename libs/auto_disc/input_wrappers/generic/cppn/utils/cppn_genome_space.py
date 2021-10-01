from auto_disc.utils.spaces import BaseSpace
from auto_disc.input_wrappers.generic.cppn import pytorchneat
import neat
from copy import deepcopy

class CPPNGenomeSpace(BaseSpace):

    def __init__(self, neat_config_filepath):
        self.neat_config = neat.Config(pytorchneat.selfconnectiongenome.SelfConnectionGenome,
                                        neat.DefaultReproduction,
                                        neat.DefaultSpeciesSet,
                                        neat.DefaultStagnation,
                                        neat_config_filepath
                                        )

        super().__init__(shape=None, dtype=None)

    def sample(self):
        genome = self.neat_config.genome_type(0)
        genome.configure_new(self.neat_config.genome_config)
        return genome

    def mutate(self, genome):
        # genome: policy_parameters.init_matnucleus_genome pytorchneat.SelfConnectionGenome
        new_genome = deepcopy(genome)
        new_genome.mutate(self.neat_config.genome_config)
        return new_genome

    def crossover(self, genome_1, genome_2):
        genome_1 = deepcopy(genome_1)
        genome_2 = deepcopy(genome_2)
        if genome_1.fitness is None:
            genome_1.fitness = 0.0
        if genome_2.fitness is None:
            genome_2.fitness = 0.0
        child_1 = self.neat_config.genome_type(0)
        child_1.configure_crossover(genome_1, genome_2, self.neat_config.genome_config)
        child_2 = self.neat_config.genome_type(0)
        child_2.configure_crossover(genome_1, genome_2, self.neat_config.genome_config)
        return child_1, child_2

    def contains(self, x):
        # TODO
        return True

    def clamp(self, x):
        # TODO
        return x

    def calc_distance(self, x1, x2):
        return x1.distance(x2, self.neat_config.genome_config)

    def expand(self, x):
        # TODO
        return