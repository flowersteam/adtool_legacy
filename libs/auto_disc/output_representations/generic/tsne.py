from auto_disc.output_representations import BaseOutputRepresentation
import torch
from auto_disc.utils.config_parameters import StringConfigParameter, DecimalConfigParameter, IntegerConfigParameter
from auto_disc.utils.spaces.utils import distance, ConfigParameterBinding
from auto_disc.utils.spaces import DictSpace, BoxSpace
from sklearn import manifold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import NotFittedError
import numpy as np

@IntegerConfigParameter(name="n_components", default=3, min=1)
@DecimalConfigParameter(name="perplexity", default=30.0, min=5.0, max=50.0)
@StringConfigParameter(name="init", default="random", possible_values=["random", "pca"])
@IntegerConfigParameter(name="fit_period", default=10, min=1)

class TSNE(BaseOutputRepresentation):
    '''
    PCA OutputRepresentation.
    '''
    output_space = DictSpace(
        embedding=BoxSpace(low=0, high=0, shape=(ConfigParameterBinding("n_components"),))
    )

    def __init__(self, wrapped_input_space_key=None):
        super().__init__(wrapped_input_space_key=wrapped_input_space_key)

        self.algorithm = make_pipeline(StandardScaler(),
                                       manifold.TSNE(n_components=self.config.n_components,
                                                     perplexity=self.config.perplexity,
                                                     init=self.config.init))
        self.output_library = np.empty((0, self.config.n_components), )

    def initialize(self, input_space):
        super().initialize(input_space)
        input_space_flatten_shape = 1
        for s in self.input_space[self.wrapped_input_space_key].shape:
            input_space_flatten_shape *= s
        self.input_library = np.empty((0, input_space_flatten_shape), )

    def map(self, observations, **kwargs):
        input = observations.reshape(1,-1)
        try:
            output = self.algorithm.transform(input)
            output = torch.from_numpy(output).flatten()
        except NotFittedError as nfe:
            print("The PCA instance is not fitted yet, returning null embedding")
            output = torch.zeros(self.output_space["embedding"].shape, dtype=self.output_space["embedding"].dtype)
        self.archive(input, output)

        if len(self.output_library) % self.config.fit_period == 0:
            self.fit_transform()

        return output

    def archive(self, input, output):
        self.input_library = np.concatenate([self.input_library, input])
        self.output_library = np.concatenate([self.output_library, output.reshape(1,-1)])

    def fit_transform(self):
        self.output_library = self.algorithm.fit_transform(self.input_library)

    def calc_distance(self, embedding_a, embedding_b, **kwargs):
        # L2 + add regularizer to avoid dead outcomes
        dist = distance.calc_l2(embedding_a, embedding_b)
        return dist
