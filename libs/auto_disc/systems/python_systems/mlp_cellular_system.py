from addict import Dict

from auto_disc.systems.python_systems import BasePythonSystem
from auto_disc.utils.config_parameters import IntegerConfigParameter, DecimalConfigParameter, StringConfigParameter
from auto_disc.utils.spaces import BoxSpace, DictSpace
from auto_disc.utils.mutators import GaussianMutator
from auto_disc.utils.spaces.utils import ConfigParameterBinding


import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, GeneralConv

import io
import imageio
import matplotlib.pyplot as plt
from PIL import Image



class UpdateRule(GeneralConv):

    def __init__(self,
                 n_channels, out_channels,
                 in_edge_channels=None,
                 aggr="add",
                 k=10,
                 ):
        super().__init__(n_channels, out_channels,
                         in_edge_channels=in_edge_channels,
                         aggr=aggr,
                         skip_linear=False,
                         directed_msg=True,
                         heads=1,
                         attention=False,
                         attention_type="additive",
                         l2_normalize=False,
                         bias=False
                         )
        self.k = k

    def reset(self, lin_msg_weight, lin_self_weight, lin_edge_weight):
        self.lin_msg.weight = lin_msg_weight
        self.lin_self.weight = lin_self_weight
        self.lin_edge.weight = lin_edge_weight

    def forward(self, pos, x):
        edge_index = knn_graph(pos, self.k)

        x = F.linear(torch.cat([pos, x], dim=-1), self.mlp_weight, self.mlp_bias)
        output = self.propagate(edge_index, x=x)
        if self.global_bias is not None:
            output += self.global_bias

        new_pos = output[:, :pos.shape[-1]]
        new_x = output[:, pos.shape[-1]:]

        return new_pos, new_x

@IntegerConfigParameter(name="n_nodes", default=20, min=1)
@IntegerConfigParameter(name="n_spatial_dims", default=2, min=1)
@IntegerConfigParameter(name="n_feats", default=8, min=1)
@IntegerConfigParameter(name="final_step", default=10, min=1, max=1000)
@IntegerConfigParameter(name="n_neighbors", default=10, min=1, max=1000)
class MLPCellularSystem(BasePythonSystem):
    CONFIG_DEFINITION = {}

    input_space = DictSpace(
        pos0=BoxSpace(low=-1.0, high=1.0, mutator=GaussianMutator(mean=0.0, std=0.1), indpb=0.85, dtype=torch.float32,
                      shape=(ConfigParameterBinding("n_nodes"), ConfigParameterBinding("n_spatial_dims"),)),
        x0=BoxSpace(low=-1.0, high=1.0, mutator=GaussianMutator(mean=0.0, std=0.1), indpb=0.85, dtype=torch.float32,
                    shape=(ConfigParameterBinding("n_nodes"), ConfigParameterBinding("n_feats"), )),
        mlp_weight=BoxSpace(low=-0.5, high=0.5, mutator=GaussianMutator(mean=0.0, std=0.05), indpb=0.8, dtype=torch.float32,
                                    shape=(ConfigParameterBinding("n_feats")+ConfigParameterBinding("n_spatial_dims"),
                                           ConfigParameterBinding("n_feats")+ConfigParameterBinding("n_spatial_dims"), )),
        mlp_bias=BoxSpace(low=-0.0, high=0.0, mutator=GaussianMutator(mean=0.0, std=0.0), indpb=0.,dtype=torch.float32,
                                  shape=(ConfigParameterBinding("n_feats")+ConfigParameterBinding("n_spatial_dims"), )),
        global_bias=BoxSpace(low=-0.0, high=0.0, mutator=GaussianMutator(mean=0.0, std=0.0), indpb=0., dtype=torch.float32,
                            shape=(ConfigParameterBinding("n_feats") + ConfigParameterBinding("n_spatial_dims"),))
        )
    output_space = DictSpace(
        pos=BoxSpace(low=0.0, high=0.0, mutator=GaussianMutator(mean=0.0, std=0.0), indpb=0., dtype=torch.float32,
                      shape=(ConfigParameterBinding("final_step"), ConfigParameterBinding("n_nodes"), ConfigParameterBinding("n_spatial_dims"),)),
        x=BoxSpace(low=0.0, high=0.0, mutator=GaussianMutator(mean=0., std=0.), indpb=0., dtype=torch.float32,
                    shape=(ConfigParameterBinding("final_step"), ConfigParameterBinding("n_nodes"), ConfigParameterBinding("n_feats"),)),
    )
    step_output_space = DictSpace()

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.update_rule = UpdateRule(self.config.n_neighbors)

    def reset(self, run_parameters):
        # clamp parameters if not contained in space definition
        if not self.input_space.contains(run_parameters):
            run_parameters = self.input_space.clamp(run_parameters)

        self.pos = run_parameters.pos0
        self.x = run_parameters.x0
        self.update_rule.reset(run_parameters.mlp_weight, run_parameters.mlp_bias, run_parameters.global_bias)

        self.step_idx = 0
        self._observations = Dict()
        self._observations.pos = torch.ones(self.output_space["pos"].shape,
                                                    device=run_parameters.pos0.device) * float('nan')
        self._observations.pos[self.step_idx] = run_parameters.pos0
        self._observations.x = torch.ones(self.output_space["x"].shape,
                                                    device=run_parameters.x0.device) * float('nan')
        self._observations.x[self.step_idx] = run_parameters.x0


    def step(self, action=None):

        self.pos, self.x = self.update_rule(self.pos, self.x)

        self.step_idx += 1
        self._observations.pos[self.step_idx] = self.pos
        self._observations.x[self.step_idx] = self.x

        return Dict(pos=self.pos, x=self.x), 0, self.step_idx >= self.config.final_step-1, None

    def observe(self):
        return self._observations

    def render(self, mode="mp4"):
        options = {
            "node_color": "b",
            "node_size": 50,
            "with_labels": False,
            "edge_color": "gray",
            "linewidths": 0.2,
            "width": 0.3,
        }
        im_array = []

        for nodes_position in self._observations.pos:
            spatial_dims = nodes_position.shape[1]
            projection = "3d" if spatial_dims == 3 else "2d"

            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection=projection)
            ax.set_xlim([-1.3, 1.3])
            ax.set_ylim([-1.3, 1.3])
            ax.scatter(*nodes_position.t().cpu().detach(),s=2, marker='.')
            im_io = io.BytesIO()
            plt.savefig(im_io, format="png")
            plt.close(fig)
            im_array.append(Image.open(im_io))

        if mode == "mp4":
            byte_img = io.BytesIO()
            imageio.mimwrite(byte_img, im_array, 'mp4', fps=1, output_params=["-f", "mp4"])  # -r argument is fps
            return (byte_img, "mp4")
        else:
            raise NotImplementedError

    def close(self):
        pass


