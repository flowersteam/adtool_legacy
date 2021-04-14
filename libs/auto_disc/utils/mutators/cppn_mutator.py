import torch
# import pytorchneat
from copy import deepcopy
from auto_disc.utils.mutators import BaseMutator

class CppnnMutator(BaseMutator):
    """
        description    : apply mutation to the cppn
        genome         :
    """

    def __init__(self, config):
        self.config = config

    def __call__(genome):
        # genome: policy_parameters.init_matnucleus_genome pytorchneat.SelfConnectionGenome
        new_genome = deepcopy(genome)
        # new_genome.mutate(self.config)
        return new_genome