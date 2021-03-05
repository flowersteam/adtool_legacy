from libs.auto_disc.output_representations.BaseOutputRepresentation import BaseOutputRepresentation
from libs.utils.auto_disc_parameters.AutoDiscParameter import AutoDiscParameter, ConfigParameterBinding
from libs.utils.auto_disc_parameters.parameter_types import ParameterTypesEnum
from libs.utils.auto_disc_parameters.AutoDiscSpaceDefinition import AutoDiscSpaceDefinition
from libs.utils.torch_utils import roll_n
import torch
import numpy as np

class LeniaImageRepresentation(BaseOutputRepresentation):
    CONFIG_DEFINITION = [
        AutoDiscParameter(
                    name="distance_function", 
                    type=ParameterTypesEnum.get('STRING'), 
                    values_range=["L2"], 
                    default="L2"),
        AutoDiscParameter(
                    name="env_size", 
                    type=ParameterTypesEnum.get('ARRAY', dims=[2]), 
                    values_range=[1, np.inf], 
                    default=[256, 256])
    ]

    OUTPUT_SPACE_DEFINITION = [
        AutoDiscParameter(
                    name="embedding", 
                    type=ParameterTypesEnum.get('SPACE'),
                    default=AutoDiscSpaceDefinition(
                        dims=[None],
                        bounds=[0, 10], #TODO: CHANGE
                        type=ParameterTypesEnum.get('FLOAT')
                    ),
                    modifiable=False),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_latents = self.config.env_size[0] * self.config.env_size[1]
        self.output_space["embedding"].dims = [self.n_latents]


    def map(self, observations):
        """
            Maps the observations of a system to an embedding vector
            Return a torch tensor
        """
        # filter low values
        filtered_im = torch.where(observations.states[-1] > 0.2, observations.states[-1], torch.zeros_like(observations.states[-1]))

        # recenter
        mu_0 = filtered_im.sum()
        if mu_0.item() > 0:

            # implementation of meshgrid in torch
            x = torch.arange(self.config.env_size[0])
            y = torch.arange(self.config.env_size[1])
            xx = x.repeat(self.config.env_size[1], 1)
            yy = y.view(-1, 1).repeat(1, self.config.env_size[0])
            X = (xx - int(self.config.env_size[0] / 2)).float()
            Y = (yy - int(self.config.env_size[1] / 2)).float()

            centroid_x = ((X * filtered_im).sum() / mu_0).round().int().item()
            centroid_y = ((Y * filtered_im).sum() / mu_0).round().int().item()

            filtered_im = roll_n(filtered_im, 0, centroid_x)
            filtered_im = roll_n(filtered_im, 1, centroid_y)

        embedding = filtered_im.flatten()

        return embedding


    def calc_distance(self, embedding_a, embedding_b):
        """
            Compute the distance between 2 embeddings in the latent space
            /!\ batch mode embedding_a and embedding_b can be N*M or M
        """
        # l2 loss
        if self.config.distance_function == "L2":
            dist = (embedding_a - embedding_b).pow(2).sum(-1).sqrt() - 0.5 * embedding_b.pow(2).sum(-1).sqrt() # add regularizer to avoid dead outcomes

        else:
            raise NotImplementedError

        return dist