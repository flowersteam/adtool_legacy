from addict import Dict

from auto_disc.systems.python_systems import BasePythonSystem
from auto_disc.utils.config_parameters import IntegerConfigParameter, DecimalConfigParameter, StringConfigParameter
from auto_disc.utils.spaces import BoxSpace, DictSpace, DiscreteSpace, MultiDiscreteSpace
from auto_disc.utils.mutators import GaussianMutator
from auto_disc.utils.spaces.utils import ConfigParameterBinding

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, MessagePassing

import io
import imageio
import matplotlib.pyplot as plt
from matplotlib import colors
from PIL import Image


# Lenia family of functions for the kernel K and for the growth mapping g
kernel_core = {
    0: lambda x,r,w,b: (b*torch.exp(-((x.unsqueeze(-1)-r)/w)**2 / 2)).sum(-1)
}

field_func = {
    0: lambda n, m, s: torch.max(torch.zeros_like(n), 1 - (n - m) ** 2 / (9 * s ** 2)) ** 4 * 2 - 1, # polynomial (quad4)
    1: lambda n, m, s: torch.exp(- (n - m) ** 2 / (2 * s ** 2)-1e-3) * 2 - 1,  # exponential / gaussian (gaus)
    2: lambda n, m, s: (torch.abs(n - m) <= s).float() * 2 - 1 , # step (stpz)
    3: lambda n, m, s: - torch.clamp(n-m,0,1)*s #food eating kernl
}

class LeniaStepConv(torch.nn.Module):

    def __init__(self, aggr="mean", nb_neighbors=None, nb_channels=1):
        super().__init__()
        self.aggr = aggr
        self.nb_neighbors = nb_neighbors #fix number of neighbors for v0 and max number of neighborhs in radius for v1
        self.nb_channels = nb_channels

    def reset(self, R, T, c0, c1, r, rk, b, w, h, m, s, kn=0, gn=1, is_soft_clip=False):
        self.R = R+15 #quick fix that allow to have R between 15 and self.output_space.R.n
        self.T = T
        self.c0 = c0
        self.c1 = c1
        self.r = r
        self.rk = rk
        self.b = b
        self.w = w
        self.h = h
        self.m = m
        self.s = s

        self.kn = kn
        self.gn = gn
        self.nb_rules = c0.shape[0]
        self.is_soft_clip = is_soft_clip

    def forward(self, pos, x):
        # v0 with grid structure
        assert self.nb_neighbors == int(1+8*(self.R+1)*(self.R+1+1)/2) # 1+sum(8r)_{r=1..R+1} (R+1 too have some margin)
        flat_indices = torch.arange(0, pos.shape[0])
        grid_side = int(np.ceil(np.power(pos.shape[0], 1 / pos.shape[1])))
        edge_index = torch.zeros((2, self.nb_neighbors * pos.shape[0]), dtype=torch.int64, device=pos.device)
        edge_index[1] = torch.repeat_interleave(flat_indices, self.nb_neighbors)
        i_indices = flat_indices // grid_side
        j_indices = flat_indices % grid_side
        shift_index = 0
        for shift_i in range(-self.R, self.R+1):
            for shift_j in range(-self.R, self.R+1):
                shifted_i_indices = (i_indices + shift_i) % grid_side
                shifted_j_indices = (j_indices + shift_j) % grid_side
                shifted_flat_indices = grid_side*shifted_i_indices + shifted_j_indices
                edge_index[0, shift_index::self.nb_neighbors] = shifted_flat_indices
                shift_index += 1

        x_extended = torch.cat([x.unsqueeze(1), x[edge_index[0]].reshape(x.shape[0], self.nb_neighbors, x.shape[1])], dim=1)  # N,nb_neighbors+1,nb_channels

        delta_pos = torch.cat([torch.zeros_like(pos).unsqueeze(1),
                               (pos[edge_index[0]] - pos[edge_index[1]]).reshape(pos.shape[0], self.nb_neighbors,
                                                                                 pos.shape[1])
                               ], dim=1)  # N,nb_neighbors+1,nb_spatial_dims

        distance = torch.sqrt(torch.sum(torch.pow(delta_pos[grid_side*grid_side//2+grid_side//2], 2), dim=-1)).unsqueeze(0).repeat(pos.shape[0], 1) / self.R

        # v1 with sparse tensors and alive filter
        # edge_index = knn_graph(pos, self.nb_neighbors)
        # edge_index = radius_graph(pos, self.R / (grid_side / 2), max_neighbors=self.nb_neighbors)
        #distance = torch.sqrt(torch.sum(torch.pow(delta_pos, 2), dim=-1))  # N,nb_neighbors+1



        kfunc = kernel_core[self.kn]
        gfunc = field_func[self.gn]
        aggr_func = eval(f"torch.{self.aggr}")

        delta_x = torch.zeros_like(x)

        for k in range(self.nb_rules):
            weight = torch.sigmoid(-(distance - 1) * 10) * kfunc(distance / self.r[k], self.rk[k], self.w[k], self.b[k])
            weight_sum = torch.sum(weight, 1).unsqueeze(1)
            weight_norm = (weight / weight_sum)
            potential = aggr_func(weight_norm * x_extended[:, :, self.c0[k]], dim=1)
            field = gfunc(potential, self.m[k], self.s[k])
            delta_x[:, self.c1[k]] = delta_x[:, self.c1[k]] + self.h[k] * field

        new_x = torch.clamp(x + (1.0 / self.T) * delta_x, min=0., max=1.)
        new_pos = pos #TODO

        return new_pos, new_x

@IntegerConfigParameter(name="nb_nodes", default=20, min=1)
@IntegerConfigParameter(name="nb_dims", default=2, min=1)
@IntegerConfigParameter(name="nb_neighbors", default=10, min=1, max=100000)
@IntegerConfigParameter(name="nb_rules", default=10, min=1)
@IntegerConfigParameter(name="nb_channels", default=1, min=1)
@StringConfigParameter(name="aggr", possible_values=["sum", "mean", "max", "min"], default="sum")
@IntegerConfigParameter(name="final_step", default=10, min=1, max=1000)
class LeniaCellularSystem(BasePythonSystem):
    CONFIG_DEFINITION = {}

    input_space = DictSpace(
        pos0=BoxSpace(low=0., high=1000.0, mutator=GaussianMutator(mean=0.0, std=0.1), indpb=0.85, dtype=torch.float32,
                      shape=(ConfigParameterBinding("nb_nodes"), ConfigParameterBinding("nb_dims"),)),
        x0=BoxSpace(low=0., high=1000.0, mutator=GaussianMutator(mean=0.0, std=0.1), indpb=0.85, dtype=torch.float32,
                    shape=(ConfigParameterBinding("nb_nodes"), ConfigParameterBinding("nb_channels"), )),
        R=DiscreteSpace(n=25, mutator=GaussianMutator(mean=0.0, std=0.01), indpb=0.01),
        T=BoxSpace(low=1.0, high=10.0, mutator=GaussianMutator(mean=0.0, std=0.1), shape=(), indpb=0.01,
                   dtype=torch.float32),
        c0=MultiDiscreteSpace(nvec=[1] * 10, mutator=GaussianMutator(mean=0.0, std=0.1), indpb=0.1),
        c1=MultiDiscreteSpace(nvec=[1] * 10, mutator=GaussianMutator(mean=0.0, std=0.1), indpb=0.1),
        rk=BoxSpace(low=0, high=1, shape=(ConfigParameterBinding("nb_rules"), 3),
                    mutator=GaussianMutator(mean=0.0, std=0.2), indpb=0.25, dtype=torch.float32),
        b=BoxSpace(low=1e-3, high=1.0, shape=(ConfigParameterBinding("nb_rules"), 3),
                   mutator=GaussianMutator(mean=0.0, std=0.2), indpb=0.25, dtype=torch.float32),
        w=BoxSpace(low=1e-3, high=0.5, shape=(ConfigParameterBinding("nb_rules"), 3),
                   mutator=GaussianMutator(mean=0.0, std=0.2), indpb=0.25, dtype=torch.float32),
        m=BoxSpace(low=0.05, high=0.5, shape=(ConfigParameterBinding("nb_rules"),),
                   mutator=GaussianMutator(mean=0.0, std=0.2), indpb=0.25, dtype=torch.float32),
        s=BoxSpace(low=1e-3, high=0.18, shape=(ConfigParameterBinding("nb_rules"),),
                   mutator=GaussianMutator(mean=0.0, std=0.01), indpb=0.25, dtype=torch.float32),
        h=BoxSpace(low=0, high=1.0, shape=(ConfigParameterBinding("nb_rules"),), mutator=GaussianMutator(mean=0.0, std=0.2),
                   indpb=0.25, dtype=torch.float32),
        r=BoxSpace(low=0.2, high=1.0, shape=(ConfigParameterBinding("nb_rules"),),
                   mutator=GaussianMutator(mean=0.0, std=0.2), indpb=0.25, dtype=torch.float32)
        # kn = DiscreteSpace(n=4, mutator=GaussianMutator(mean=0.0, std=0.1), indpb=1.0),
        # gn = DiscreteSpace(n=3, mutator=GaussianMutator(mean=0.0, std=0.1), indpb=1.0),
        )
    output_space = DictSpace(
        pos=BoxSpace(low=0.0, high=0.0, mutator=GaussianMutator(mean=0.0, std=0.0), indpb=0., dtype=torch.float32,
                      shape=(ConfigParameterBinding("final_step"), ConfigParameterBinding("nb_nodes"), ConfigParameterBinding("nb_dims"),)),
        x=BoxSpace(low=0.0, high=0.0, mutator=GaussianMutator(mean=0., std=0.), indpb=0., dtype=torch.float32,
                    shape=(ConfigParameterBinding("final_step"), ConfigParameterBinding("nb_nodes"), ConfigParameterBinding("nb_channels"),)),
    )
    step_output_space = DictSpace()

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.input_space.spaces["c0"] = MultiDiscreteSpace(nvec=[self.config.nb_channels] * self.config.nb_rules,
                                                           mutator=GaussianMutator(mean=0.0, std=0.1), indpb=0.1)
        self.input_space.spaces["c0"].initialize(self)

        self.input_space.spaces["c1"] = MultiDiscreteSpace(nvec=[self.config.nb_channels] * self.config.nb_rules,
                                                           mutator=GaussianMutator(mean=0.0, std=0.1), indpb=0.1)
        self.input_space.spaces["c1"].initialize(self)
        
        self.update_rule = LeniaStepConv(nb_neighbors=self.config.nb_neighbors, aggr=self.config.aggr)

    def reset(self, run_parameters):
        # clamp parameters if not contained in space definition
        if not self.input_space.contains(run_parameters):
            run_parameters = self.input_space.clamp(run_parameters)

        self.pos = run_parameters.pos0
        self.x = run_parameters.x0
        self.update_rule.reset(R=run_parameters.R, T=run_parameters.T,
                               c0=run_parameters.c0, c1=run_parameters.c1,
                               r=run_parameters.r, rk=run_parameters.rk,
                               b=run_parameters.b, w=run_parameters.w, h=run_parameters.h,
                               m=run_parameters.m, s=run_parameters.s)

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

        channel_colors = [colors.to_rgb(color) for color in colors.TABLEAU_COLORS.values()][:self.config.nb_channels]
        channel_colors = torch.as_tensor(channel_colors).t()
        for nodes_pos, nodes_x in zip(self._observations.pos, self._observations.x):
            nodes_pos = nodes_pos.detach().cpu().t()
            spatial_dims = nodes_pos.shape[0]
            grid_side = int(np.ceil(np.power(nodes_pos.shape[1], 1 / nodes_pos.shape[0])))
            projection = "3d" if spatial_dims == 3 else "rectilinear"
            nodes_color = (channel_colors@nodes_x.detach().cpu().t()).t()
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection=projection)
            ax.set_xlim([nodes_pos[0].min().item(), nodes_pos[0].max().item()])
            ax.set_ylim([nodes_pos[1].min().item(), nodes_pos[1].max().item()])

            # retrieve the pixel information:
            xy_pixels = ax.transData.transform(nodes_pos.t())
            xpix, ypix = xy_pixels.T
            # this assumes that your data-points are equally spaced
            s = xpix[grid_side] - xpix[0]
            ax.scatter(nodes_pos[1], grid_side - nodes_pos[0], s=s**2, marker='s', c=nodes_color) #ordering matplotlib
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


