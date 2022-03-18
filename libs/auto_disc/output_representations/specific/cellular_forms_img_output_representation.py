from auto_disc.output_representations import BaseOutputRepresentation
from auto_disc.utils.config_parameters import StringConfigParameter, DecimalConfigParameter, IntegerConfigParameter
from auto_disc.utils.spaces.utils import ConfigParameterBinding
from auto_disc.utils.spaces import DictSpace, BoxSpace, DiscreteSpace
from auto_disc.utils.misc.torch_utils import roll_n
from auto_disc.utils.spaces.utils import distance

import torch
import torchvision.transforms as TF
from torchvision.models import resnet18

from addict import Dict

resize = TF.Resize(244)

@IntegerConfigParameter(name="resolution", default=1024)
@StringConfigParameter(name="distance_function", default="L2")
class CellularFormsResNet18OutputRepresentation(BaseOutputRepresentation):
    """
    Image-based BC for the CellularForms system, based on the last layer of a
    ResNet18 model pretrained on ImageNet.
    """

    CONFIG_DEFINITION = {}
    config = Dict()

    output_space = DictSpace(
        embedding=BoxSpace(low=0, high=10, shape=(512,))
    )

    def __init__(self, wrapped_input_space_key=None):
        super().__init__('states')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.encoder = resnet18(pretrained=True)
        self.encoder.to(self.device)
        self.embedding = None

        def hook(model, input, output):
            self.embedding = output.squeeze().detach()

        self.encoder.avgpool.register_forward_hook(hook)

    def map(self, observations, is_output_new_discovery):
        """
            Maps the observations of a system to an embedding vector
            Return a torch tensor
        """
        last_image = torch.stack([observations.states[-1]] * 3, dim=0).unsqueeze(0).to(self.device)
        last_image = resize(last_image)
        _ = self.encoder(last_image)
        return {'embedding': self.embedding.cpu()}


    def calc_distance(self, embedding_a, embedding_b, **kwargs):
        """
            Compute the distance between 2 embeddings in the latent space
            /!\ batch mode embedding_a and embedding_b can be N*M or M
        """
        # l2 loss
        if self.config.distance_function == "L2":
            dist = distance.calc_l2(embedding_a, embedding_b) # add regularizer to avoid dead outcomes

        else:
            raise NotImplementedError

        return dist