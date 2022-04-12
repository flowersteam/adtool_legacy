from auto_disc.output_representations import BaseOutputRepresentation
import torch
from auto_disc.utils.config_parameters import StringConfigParameter, DecimalConfigParameter, IntegerConfigParameter, BooleanConfigParameter
from auto_disc.utils.spaces.utils import ConfigParameterBinding
from auto_disc.utils.spaces import DictSpace, BoxSpace
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import NotFittedError
import numpy as np

@IntegerConfigParameter(name="n_components", default=3, min=1)
@IntegerConfigParameter(name="n_neighbors", default=15, min=2, max=200)
@DecimalConfigParameter(name="min_dist", default=0.1, min=0.0, max=0.99)
@StringConfigParameter(name="init", default="spectral", possible_values=["spectral", ])
@StringConfigParameter(name="metric", default="euclidean", possible_values=["euclidean", ])
@IntegerConfigParameter(name="fit_period", default=10, min=1)
@BooleanConfigParameter(name="expand_output_space", default=True)

class UMAP(BaseOutputRepresentation):
    '''
    UMAP OutputRepresentation.
    '''
    CONFIG_DEFINITION = {}

    output_space = DictSpace(
        umap=BoxSpace(low=0, high=0, shape=(ConfigParameterBinding("n_components"),))
    )

    def __init__(self, wrapped_input_space_key=None, **kwargs):
        super().__init__(wrapped_input_space_key=wrapped_input_space_key, **kwargs)

        self.algorithm = make_pipeline(StandardScaler(),
                                       umap.UMAP(n_components=self.config.n_components,
                                                 n_neighbors=self.config.n_neighbors,
                                                 min_dist=self.config.min_dist,
                                                 init=self.config.init,
                                                 metric=self.config.metric,
                                                 ))
        # quick fix
        self.output_space[f"umap_{self.wrapped_input_space_key}"] = self.output_space.spaces.pop("umap")

    def map(self, observations, is_output_new_discovery, **kwargs):
        input = observations[self.wrapped_input_space_key].reshape(1,-1)
        try:
            output = self.algorithm.transform(input)
            output = torch.from_numpy(output).flatten()
        except NotFittedError as nfe:
            print("The UMAP instance is not fitted yet, returning null embedding")
            output = torch.zeros(self.output_space[f"umap_{self.wrapped_input_space_key}"].shape, dtype=self.output_space[f"umap_{self.wrapped_input_space_key}"].dtype)

        if (self.CURRENT_RUN_INDEX % self.config.fit_period == 0) and (self.CURRENT_RUN_INDEX > 0) and is_output_new_discovery:
            self.fit_update()

        output = {f"umap_{self.wrapped_input_space_key}": output}

        if self.config.expand_output_space:
            self.output_space.expand(output)

        return output

    def fit_update(self):
        input_library = np.stack([self._access_history(index=i)[0]['input'][self.wrapped_input_space_key].flatten() for i in range(self.CURRENT_RUN_INDEX-1)])
        self.algorithm.fit(input_library)
        self._call_output_history_update()


