from addict import Dict
from auto_disc.systems.python_systems import BasePythonSystem
from auto_disc.utils.spaces import BoxSpace, DictSpace
from auto_disc.utils.config_parameters import IntegerConfigParameter, DecimalConfigParameter, StringConfigParameter
from auto_disc.utils.mutators import GaussianMutator
from copy import deepcopy
import math
import torch
from torch import nn

import io
import imageio
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image

class Node():

    def __init__(self, parent=None, net=None, x=None):

        # some tests
        if x is not None:
            assert (parent is None and net is None), ("Root nodes must have no"
                                                      " parent and no provided net, set these to None")
            x = torch.tensor(x)
        else:
            assert (parent is not None and net is not None), ("Intermediate "
                                                              "nodes must have a parent and a net")

        self.parent = parent
        self.net = net

        if self.parent is None:
            self.depth = 0
            self.x = x
        else:
            self.parent.children.append(self)
            self.depth = self.parent.depth + 1
            self.x = self.net(self.parent.x)

        self.children = []

class UpdateRule(nn.Module):

    def __init__(self, r, theta, direction):
        super().__init__()
        self.r = r
        self.theta = theta
        self.direction = direction

    def forward(self, x):
        cur_r = self.r[0]
        self.r = self.r.roll(-1)

        cur_theta = self.theta[0]
        self.theta = self.theta.roll(-1)

        xpos, ypos, angle = x

        angle = angle + self.direction * cur_theta
        xpos = xpos + (cur_r * torch.cos(angle))
        ypos = ypos + (cur_r * torch.sin(angle))

        return torch.stack([xpos, ypos, angle])


@DecimalConfigParameter("r_min", default=0.05)
@DecimalConfigParameter("r_max", default=0.4)
@DecimalConfigParameter("theta_min", default=0.1)
@DecimalConfigParameter("theta_max", default=torch.acos(torch.zeros(1)).item())
@IntegerConfigParameter("tree_depth", default=3, min=1, max=10)
@StringConfigParameter("x_root_init", default="torch.zeros(3)")
class ToyCellularSystem (BasePythonSystem):
    CONFIG_DEFINITION = {}

    input_space = DictSpace(
        r=BoxSpace(low=0.0, high=0.0, mutator=GaussianMutator(mean=0.0, std=0.0), indpb=0.0, dtype=torch.float32, shape=()),
        theta=BoxSpace(low=0.0, high=0.0, mutator=GaussianMutator(mean=0.0, std=0.0), indpb=0.0, dtype=torch.float32, shape=())
    )
    output_space = DictSpace(
        node_features=BoxSpace(low=0.0, high=1.0, mutator=GaussianMutator(mean=0.0, std=0.0), indpb=0.0, dtype=torch.float32, shape=()),
    )
    step_output_space = DictSpace()

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.input_space["r"] = BoxSpace(low=self.config.r_min, high=self.config.r_max,
                                         mutator=GaussianMutator(mean=0.0, std=0.15*(self.config.r_max-self.config.r_min)), indpb=1.0,
                                         dtype=torch.float32, shape=(math.pow(2,self.config.tree_depth) - 1, ))
        self.input_space["r"].initialize(self)
        self.input_space["theta"] = BoxSpace(low=self.config.theta_min, high=self.config.theta_max,
                                             mutator=GaussianMutator(mean=0.0, std=0.15*(self.config.theta_max-self.config.theta_min)), indpb=1.0,
                                             dtype=torch.float32, shape=(2*(math.pow(2,self.config.tree_depth) - 1), ))
        self.input_space["theta"].initialize(self)

        self.output_space["node_features"] = BoxSpace(low=0.0,
                                                   high=0.0,
                                                   mutator=GaussianMutator(mean=0.0, std=0.1 * (
                                                         self.config.r_max - self.config.r_min)), indpb=1.0,
                                                   dtype=torch.float32, shape=(self.config.tree_depth+1, math.pow(2, self.config.tree_depth+1)-1, 2))
        self.output_space["node_features"].initialize(self)


    def reset(self, run_parameters):

        # clamp parameters if not contained in space definition
        if not self.input_space.contains(run_parameters):
            run_parameters = self.input_space.clamp(run_parameters)

        self.root = Node(x=eval(self.config.x_root_init).to(run_parameters.r.device))
        self.fun_plus = UpdateRule(run_parameters.r, run_parameters.theta[:7], 1.0)
        self.fun_minus = UpdateRule(run_parameters.r, run_parameters.theta[-7:], -1.0)

        self.node_list = [self.root]
        self.edge_list = []

        self.step_idx = 0
        self._observations = Dict()
        self._observations.node_features = torch.ones(self.output_space["node_features"].shape,
                                                    device=run_parameters.r.device) * float('nan')
        self._observations.node_features[self.step_idx, :len(self.node_list), :] = torch.cat([deepcopy(node.x[:2].unsqueeze(0)) for node in self.node_list])


    def step(self, action=None):

        for node_idx, node in enumerate(self.node_list):
            if node.depth == self.step_idx:
                node_plus = Node(parent=node, net=self.fun_plus)
                node_minus = Node(parent=node, net=self.fun_minus)
                self.node_list += [node_plus, node_minus]
                self.edge_list += [(node_idx, len(self.node_list)-2), (node_idx, len(self.node_list)-1)]

        self.step_idx += 1
        self._observations.node_features[self.step_idx, :len(self.node_list), :] = torch.cat(
            [deepcopy(node.x[:2].unsqueeze(0)) for node in self.node_list])

        return Dict(node_list=self.node_list, edge_list=self.edge_list), 0, self.step_idx == self.config.tree_depth, None

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

        for node_feats in self._observations.node_features:
            G = nx.DiGraph()
            pos_dict = {}
            n_nodes = 0
            for node_idx, node_feat in enumerate(node_feats):
                if torch.isnan(node_feat).all():
                    break
                n_nodes += 1
                G.add_node(node_idx)
                pos_dict[node_idx] = node_feat.tolist()
            for edge in self.edge_list[:n_nodes-1]:
                G.add_edge(*edge)


            fig = plt.figure(figsize=(4,4))

            ax = fig.add_subplot(111)
            ax.set_xlim([-1.3, 1.3])
            ax.set_ylim([-1.3, 1.3])
            nx.draw(G, pos=pos_dict, **options)
            im_io = io.BytesIO()
            plt.savefig(im_io, format="png")
            im_array.append(Image.open(im_io))

        if mode == "mp4":
            byte_img = io.BytesIO()
            imageio.mimwrite(byte_img, im_array, 'mp4', fps=1, output_params=["-f", "mp4"]) #-r argument is fps
            return (byte_img, "mp4")
        else:
            raise NotImplementedError

    def close(self):
        pass