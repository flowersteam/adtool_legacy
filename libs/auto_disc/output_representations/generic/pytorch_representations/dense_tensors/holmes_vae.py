from addict import Dict
from auto_disc.output_representations import BaseOutputRepresentation
from auto_disc.output_representations.generic.pytorch_representations.dense_tensors.trunks.encoders import get_encoder, ConnectedEncoder
from auto_disc.output_representations.generic.pytorch_representations.dense_tensors.heads.decoders import get_decoder, ConnectedDecoder
from auto_disc.output_representations.generic.pytorch_representations.dense_tensors.vae import VAELoss
from auto_disc.utils.config_parameters import IntegerConfigParameter, StringConfigParameter, BooleanConfigParameter
from auto_disc.utils.misc.dict_utils import map_nested_dicts
from auto_disc.utils.misc.torch_utils import ExperimentHistoryDataset, ModelWrapper
from auto_disc.utils.misc.tensorboard_utils import logger_add_image_list, resize_embeddings
from auto_disc.utils.spaces.utils import ConfigParameterBinding
from auto_disc.utils.spaces import DictSpace, BoxSpace
from copy import deepcopy
import math
import numpy as np
import os
from sklearn import cluster, svm, mixture
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import sys
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


"""=======================================================================================
Node base class
========================================================================================="""
class Node(nn.Module):
    """
    Node base class
    """

    @staticmethod
    def path_to_id(path):
        if len(path) == 1:
            return int(1)
        else:
            return 2 * Node.path_to_id(path[:-1]) + int(path[-1])


    @staticmethod
    def id_to_path(id):
        if id == 1:
            return "0"
        else:
            return Node.id_to_path(id // 2) + str(id % 2)

    def __init__(self, path, **kwargs):
        self.path = path
        self.depth = len(path) - 1
        self.id = Node.path_to_id(self.path)
        self.leaf = True  # set to False when node is split
        self.boundary = None
        self.leaf_accumulator = []
        self.fitness_last_epochs = []

    def reset_accumulator(self):
        self.leaf_accumulator = []
        if not self.leaf:
            self.left.reset_accumulator()
            self.right.reset_accumulator()

    def get_child_node(self, path):
        node = self
        for d in range(1, len(path)):
            if path[d] == "0":
                node = node.left
            else:
                node = node.right

        return node

    def get_leaf_pathes(self, path_taken=[]):
        if self.depth == 0:
            path_taken = "0"
        if self.leaf:
            return ([path_taken])

        else:
            left_leaf_accumulator = self.left.get_leaf_pathes(path_taken=path_taken + "0")
            self.leaf_accumulator.extend(left_leaf_accumulator)
            right_leaf_accumulator = self.right.get_leaf_pathes(path_taken=path_taken + "1")
            self.leaf_accumulator.extend(right_leaf_accumulator)
            leaf_accumulator = self.leaf_accumulator
            self.reset_accumulator()

        return leaf_accumulator

    def get_node_pathes(self, path_taken=[]):
        if self.depth == 0:
            path_taken = "0"
        if self.leaf:
            return ([path_taken])

        else:
            self.leaf_accumulator.extend([path_taken])
            left_leaf_accumulator = self.left.get_node_pathes(path_taken=path_taken + "0")
            self.leaf_accumulator.extend(left_leaf_accumulator)
            right_leaf_accumulator = self.right.get_node_pathes(path_taken=path_taken + "1")
            self.leaf_accumulator.extend(right_leaf_accumulator)
            leaf_accumulator = self.leaf_accumulator
            self.reset_accumulator()

        return leaf_accumulator

    def create_boundary(self, z_library, z_fitness=None, boundary_name="cluster.KMeans", boundary_parameters={}):
        X = z_library
        boundary_algo = eval(boundary_name)

        if z_fitness is None:
            if boundary_name == "cluster.KMeans":
                boundary_parameters.n_clusters = 2
                self.boundary = make_pipeline(StandardScaler(),
                                      boundary_algo(**boundary_parameters))
                self.boundary.fit(X)
        else:
            y = z_fitness.squeeze()
            if boundary_name == 'cluster.KMeans':
                center0 = np.median(X[y <= np.percentile(y, 20), :], axis=0)
                center1 = np.median(X[y > np.percentile(y, 80), :], axis=0)
                center = np.stack([center0, center1])
                center = np.nan_to_num(center)
                boundary_parameters.init = center
                boundary_parameters.n_clusters = 2
                self.boundary = make_pipeline(StandardScaler(),
                                              boundary_algo(**boundary_parameters))
                self.boundary.fit(X)
            elif boundary_name == 'svm.SVC':
                y = y > np.percentile(y, 80)
                self.boundary = make_pipeline(StandardScaler(),
                                              boundary_algo(**boundary_parameters))
                self.boundary.fit(X,y)



        return


    def depth_first_forward(self, x, tree_path_taken=None, x_ids=None, parent_lf=None, parent_gf=None,
                            parent_gfi=None, parent_lfi=None, parent_recon_x=None):
        if self.depth == 0:
            tree_path_taken = ["0"] * x.shape[0]  # all the paths start with "O"
            x_ids = list(range(x.shape[0]))

        node_outputs = self.node_forward(x, parent_lf, parent_gf, parent_gfi, parent_lfi, parent_recon_x)

        parent_lf = node_outputs["lf"].detach()
        parent_gf = node_outputs["gf"].detach()
        parent_gfi = node_outputs["gfi"].detach()
        parent_lfi = node_outputs["lfi"].detach()
        parent_recon_x = node_outputs["recon_x"].detach()

        if self.leaf:
            # return path_taken, x_ids, leaf_outputs
            return ([[tree_path_taken, x_ids, node_outputs]])

        else:
            z = node_outputs["z"]
            x_side = self.get_children_node(z)
            x_ids_left = np.where(x_side == 0)[0]
            if len(x_ids_left) > 0:
                left_leaf_accumulator = self.left.depth_first_forward(x[x_ids_left], [path + "0" for path in
                                                                                      [tree_path_taken[x_idx] for x_idx
                                                                                       in x_ids_left]],
                                                                      [x_ids[x_idx] for x_idx in x_ids_left],
                                                                      parent_lf[x_ids_left],
                                                                      parent_gf[x_ids_left],
                                                                      parent_gfi[x_ids_left],
                                                                      parent_lfi[x_ids_left],
                                                                      parent_recon_x[x_ids_left])
                self.leaf_accumulator.extend(left_leaf_accumulator)

            x_ids_right = np.where(x_side == 1)[0]
            if len(x_ids_right) > 0:
                right_leaf_accumulator = self.right.depth_first_forward(x[x_ids_right], [path + "1" for path in
                                                                                         [tree_path_taken[x_idx] for
                                                                                          x_idx in x_ids_right]],
                                                                        [x_ids[x_idx] for x_idx in x_ids_right],
                                                                        parent_lf[x_ids_right],
                                                                        parent_gf[x_ids_right],
                                                                        parent_gfi[x_ids_right],
                                                                        parent_lfi[x_ids_right],
                                                                        parent_recon_x[x_ids_right])
                self.leaf_accumulator.extend(right_leaf_accumulator)

            leaf_accumulator = self.leaf_accumulator
            self.reset_accumulator()

        return leaf_accumulator

    def depth_first_forward_whole_branch_preorder(self, x, tree_path_taken=None, x_ids=None, parent_lf=None, parent_gf=None,
                            parent_gfi=None, parent_lfi=None, parent_recon_x=None):
        if self.depth == 0:
            tree_path_taken = ["0"] * len(x)  # all the paths start with "O"
            x_ids = list(range(len(x)))

        node_outputs = self.node_forward(x, parent_lf, parent_gf, parent_gfi, parent_lfi, parent_recon_x)

        parent_lf = node_outputs["lf"].detach()
        parent_gf = node_outputs["gf"].detach()
        parent_gfi = node_outputs["gfi"].detach()
        parent_lfi = node_outputs["lfi"].detach()
        parent_recon_x = node_outputs["recon_x"].detach()

        if self.leaf:
            # return path_taken, x_ids, leaf_outputs
            return ([[tree_path_taken, x_ids, node_outputs]])

        else:
            self.leaf_accumulator.extend([[tree_path_taken, x_ids, node_outputs]])

            z = node_outputs["z"]
            x_side = self.get_children_node(z)
            x_ids_left = np.where(x_side == 0)[0]
            if len(x_ids_left) > 0:
                left_leaf_accumulator = self.left.depth_first_forward_whole_branch_preorder(x[x_ids_left], [path + "0" for path in
                                                                                      [tree_path_taken[x_idx] for x_idx
                                                                                       in x_ids_left]],
                                                                      [x_ids[x_idx] for x_idx in x_ids_left],
                                                                      parent_lf[x_ids_left],
                                                                      parent_gf[x_ids_left],
                                                                      parent_gfi[x_ids_left],
                                                                      parent_lfi[x_ids_left],
                                                                      parent_recon_x[x_ids_left])
                self.leaf_accumulator.extend(left_leaf_accumulator)

            x_ids_right = np.where(x_side == 1)[0]
            if len(x_ids_right) > 0:
                right_leaf_accumulator = self.right.depth_first_forward_whole_branch_preorder(x[x_ids_right], [path + "1" for path in
                                                                                         [tree_path_taken[x_idx] for
                                                                                          x_idx in x_ids_right]],
                                                                        [x_ids[x_idx] for x_idx in x_ids_right],
                                                                        parent_lf[x_ids_right],
                                                                        parent_gf[x_ids_right],
                                                                        parent_gfi[x_ids_right],
                                                                        parent_lfi[x_ids_right],
                                                                        parent_recon_x[x_ids_right])
                self.leaf_accumulator.extend(right_leaf_accumulator)

            leaf_accumulator = self.leaf_accumulator
            self.reset_accumulator()

        return leaf_accumulator


    def depth_first_forward_whole_tree_preorder(self, x, tree_path_taken=None, parent_lf=None, parent_gf=None,
                            parent_gfi=None, parent_lfi=None, parent_recon_x=None):
        if self.depth == 0:
            tree_path_taken = ["0"] * len(x)  # all the paths start with "O"

        node_outputs = self.node_forward(x, parent_lf, parent_gf, parent_gfi, parent_lfi, parent_recon_x)

        parent_lf = node_outputs["lf"].detach()
        parent_gf = node_outputs["gf"].detach()
        parent_gfi = node_outputs["gfi"].detach()
        parent_lfi = node_outputs["lfi"].detach()
        parent_recon_x = node_outputs["recon_x"].detach()

        if self.leaf:
            # return path_taken, x_ids, leaf_outputs
            return ([[tree_path_taken, node_outputs]])

        else:
            self.leaf_accumulator.extend([[tree_path_taken, node_outputs]])
            #send everything left
            left_leaf_accumulator = self.left.depth_first_forward_whole_tree_preorder(x, [path+"0" for path in tree_path_taken],
                                                                  parent_lf, parent_gf, parent_gfi, parent_lfi, parent_recon_x)
            self.leaf_accumulator.extend(left_leaf_accumulator)

            #send everything right
            right_leaf_accumulator = self.right.depth_first_forward_whole_tree_preorder(x, [path+"1" for path in tree_path_taken],
                                                                    parent_lf, parent_gf, parent_gfi, parent_lfi, parent_recon_x)
            self.leaf_accumulator.extend(right_leaf_accumulator)

            leaf_accumulator = self.leaf_accumulator
            self.reset_accumulator()

        return leaf_accumulator

    def node_forward(self, x, parent_lf=None, parent_gf=None, parent_gfi=None, parent_lfi=None,
                     parent_recon_x=None):
        if self.depth == 0:
            encoder_outputs = self.network.encoder(x)
        else:
            encoder_outputs = self.network.encoder(x, parent_lf, parent_gf)
        model_outputs = self.node_forward_from_encoder(encoder_outputs, parent_gfi, parent_lfi, parent_recon_x)
        return model_outputs

    def get_boundary_side(self, z):
        if self.boundary is None:
            raise ValueError("Boundary computation is required before calling this function")
        else:
            # compute boundary side
            if isinstance(z, torch.Tensor):
                z = z.detach().cpu().numpy()

            side = self.boundary.predict(z)  # returns 0: left, 1: right
        return side

    def get_children_node(self, z):
        """
        Return code: -1 for leaf node,  0 to send left, 1 to send right
        """
        #z = z.to(self.config.input_tensors_device)

        if self.leaf:
            return -1 * torch.ones_like(z)
        else:
            return self.get_boundary_side(z)

    def split(self, create_boundary=False, history=None, boundary_name="cluster.KMeans", boundary_parameters={}):

        self.eval()

        # Freeze parent parameters
        for param in self.parameters():
            param.requires_grad = False

        # Instanciate childrens
        self.left = VAENode(self.path + "0", parent_network=deepcopy(self.network))
        self.right = VAENode(self.path + "1", parent_network=deepcopy(self.network))
        self.leaf = False

        # Create boundary
        if create_boundary:
            z_library = torch.stack([history[i][self.path] for i in range(len(history))])
            if torch.isnan(z_library).any():
                keep_ids = ~(torch.isnan(z_library.sum(1)))
                z_library = z_library[keep_ids]

            z_fitness = None  # TODO: allow other z_fitness

            self.create_boundary(z_library.cpu().numpy(), z_fitness, boundary_name=boundary_name, boundary_parameters=boundary_parameters)

            self.log_split()

        return


    def log_split(self):
        #TODO
        return


"""=======================================================================================
VAE Node
========================================================================================="""
class VAENode(Node, nn.Module):
    def __init__(self, path, parent_network=None,
                 encoder_name="Burgess",
                 input_shape=(1, 64, 64),
                 n_latents=10,
                 n_conv_layers=4,
                 feature_layer=1,
                 hidden_channels=32,
                 hidden_dims=256,
                 conditional_type="gaussian",
                 use_attention=False,
                 create_connections_lf=True,
                 create_connections_gf=False,
                 create_connections_gfi=True,
                 create_connections_lfi=True,
                 create_connections_recon=True):
        Node.__init__(self, path)
        nn.Module.__init__(self)

        self.network = nn.Module()

        if self.depth == 0:
            encoder_cls = get_encoder(encoder_name)
            self.network.encoder = encoder_cls(input_shape=input_shape,
                                       n_latents=n_latents,
                                       n_conv_layers=n_conv_layers,
                                       feature_layer=feature_layer,
                                       hidden_channels=hidden_channels,
                                       hidden_dims=hidden_dims,
                                       conditional_type=conditional_type,
                                       use_attention=use_attention,
                                       )
            decoder_cls = get_decoder(encoder_name)
            self.network.decoder = decoder_cls(output_shape=input_shape,
                                       n_latents=n_latents,
                                       n_conv_layers=n_conv_layers,
                                       feature_layer=feature_layer,
                                       hidden_channels=hidden_channels,
                                       hidden_dims=hidden_dims,
                                       )


        # connect encoder and decoder
        else:
            self.network.encoder = ConnectedEncoder(parent_network.encoder, connect_lf=create_connections_lf,
                                                    connect_gf=create_connections_gf)
            self.network.decoder = ConnectedDecoder(parent_network.decoder, connect_gfi=create_connections_gfi,
                                                    connect_lfi=create_connections_lfi,
                                                    connect_recon=create_connections_recon)


    def node_forward_from_encoder(self, encoder_outputs, parent_gfi=None, parent_lfi=None,
                                  parent_recon_x=None):
        if self.depth == 0:
            decoder_outputs = self.network.decoder(encoder_outputs["z"])
        else:
            decoder_outputs = self.network.decoder(encoder_outputs["z"], parent_gfi, parent_lfi,
                                                   parent_recon_x)
        model_outputs = encoder_outputs
        model_outputs.update(decoder_outputs)
        return model_outputs

"""=======================================================================================
HOLMES CLASS 
========================================================================================="""
#@StringConfigParameter(name="input_tensors_layout", default="dense", possible_values=["dense", "minkowski", ])
@StringConfigParameter(name="input_tensors_device", default="cpu", possible_values=["cuda", "cpu", ])
#TODO: Add device as part of spaces (?)

@StringConfigParameter(name="encoder_name", default="Burgess", possible_values=["Burgess", "Dumoulin", ])
@IntegerConfigParameter(name="encoder_n_latents", default=10, min=1)
@IntegerConfigParameter(name="encoder_n_conv_layers", default=4, min=1)
@IntegerConfigParameter(name="encoder_feature_layer", default=1, min=1)
@IntegerConfigParameter(name="encoder_hidden_channels", default=32, min=1)
@IntegerConfigParameter(name="encoder_hidden_dims", default=256, min=1)
@StringConfigParameter(name="encoder_conditional_type", default="gaussian", possible_values=["gaussian", "deterministic", ])
@BooleanConfigParameter(name="encoder_use_attention", default=False)

@StringConfigParameter(name="weights_init_name", default="pytorch", possible_values=["pytorch", ])
#TODO: DictConfigParameter for weights_init_parameters

@StringConfigParameter(name="loss_name", default="VAE", possible_values=["VAE", "betaVAE", "annealedVAE", ])
#TODO: DictConfigParameter for loss_parameters

@StringConfigParameter(name="optimizer_name", default="Adam", possible_values=["Adam"])
#TODO: DictConfigParameter for optimizer_parameters

#TODO: proper save function with AutoDiscTool

#@BooleanConfigParameter(name="tensorboard_active", default=True)
# TODO: replace tensorboard frequency with callbacks
@StringConfigParameter(name="tb_folder", default="./tensorboard", possible_values="all")
@IntegerConfigParameter(name="tb_record_loss_frequency", default=1, min=1)
@IntegerConfigParameter(name="tb_record_images_frequency", default=10, min=1)
@IntegerConfigParameter(name="tb_record_embeddings_frequency", default=10, min=1)
@IntegerConfigParameter(name="tb_record_memory_max", default=100, min=1) #TODO: do something better for this

@BooleanConfigParameter(name="create_connections_lf", default=True)
@BooleanConfigParameter(name="create_connections_gf", default=False)
@BooleanConfigParameter(name="create_connections_gfi", default=True)
@BooleanConfigParameter(name="create_connections_lfi", default=True)
@BooleanConfigParameter(name="create_connections_recon", default=True)


#TODO: think of HOLMES output space form: specific sample() and calc_distance()
#TODO: save model before/after split callbacks

@BooleanConfigParameter(name="split_active", default=False)
@StringConfigParameter(name="split_loss_key", default="recon", possible_values=["total", "recon"])
@StringConfigParameter(name="split_type", default="plateau", possible_values=["plateau", "threshold", ])
@IntegerConfigParameter(name="split_parameters_epsilon", default=1)
@IntegerConfigParameter(name="split_parameters_n_steps_average", default=5)
@IntegerConfigParameter(name="n_min_epochs_before_split", default=1)
@IntegerConfigParameter(name="n_min_epochs_between_splits", default=1)
@IntegerConfigParameter(name="n_min_points_for_split", default=1)
@IntegerConfigParameter(name="n_max_splits", default=1)

@StringConfigParameter(name="boundary_name", default="cluster.KMeans", possible_values=["cluster.KMeans","svm.SVC"])
#TODO: DictConfigParameter for boundary_parameters

@IntegerConfigParameter(name="train_period", default=20, min=1)
@IntegerConfigParameter(name="n_epochs_per_train_period", default=20, min=1)
@BooleanConfigParameter(name="alternated_backward_active", default=False)
@IntegerConfigParameter(name="alternated_backward_period", default=10, min=1)
@IntegerConfigParameter(name="alternated_backward_connections", default=1, min=1)


@IntegerConfigParameter(name="dataloader_batch_size", default=10, min=1)
@IntegerConfigParameter(name="dataloader_num_workers", default=0, min=0)
@BooleanConfigParameter(name="dataloader_drop_last", default=True)

@BooleanConfigParameter(name="expand_output_space", default=True)

class HOLMES_VAE(nn.Module, BaseOutputRepresentation):
    """
    HOLMES_VAE Output Representation
    """

    output_space = DictSpace(
        spaces={"0": BoxSpace(low=0, high=0, shape=(ConfigParameterBinding("encoder_n_latents"),))}
                     )

    def __init__(self, wrapped_input_space_key=None):
        BaseOutputRepresentation.__init__(self, wrapped_input_space_key=wrapped_input_space_key)
        nn.Module.__init__(self)

    def initialize(self, input_space):
        BaseOutputRepresentation.initialize(self, input_space)

        # Model
        self.set_model()

        # Loss
        self.set_loss()

        # Optimizer
        self.set_optimizer()

        # Scheduler
        self.set_scheduler()

        # Compute
        self.set_input_tensors_compatibility()

        # Tensorboard Logger
        self.set_tensorboard()

        # set counters
        self.n_epochs = 0
        self.split_history = {}

    def set_model(self):
        """
        Instantiates the torch model weights based on self.config.encoder
        The model must have an encoder and define the input shape and output shape (of that encoder)
        :return:
        """
        self.root = VAENode(path="0",
                            encoder_name=self.config.encoder_name,
                            input_shape=self.input_space[self.wrapped_input_space_key].shape,
                            n_latents=self.config.encoder_n_latents,
                            n_conv_layers=self.config.encoder_n_conv_layers,
                            feature_layer=self.config.encoder_feature_layer,
                            hidden_channels=self.config.encoder_hidden_channels,
                            hidden_dims=self.config.encoder_hidden_dims,
                            conditional_type=self.config.encoder_conditional_type,
                            use_attention=self.config.encoder_use_attention,
                            create_connections_lf=self.config.create_connections_lf,
                            create_connections_gf=self.config.create_connections_gf,
                            create_connections_gfi=self.config.create_connections_gfi,
                            create_connections_lfi=self.config.create_connections_lfi,
                            create_connections_recon=self.config.create_connections_recon)

        self.init_network_weights()


    def init_network_weights(self):
        #TODO
        return

    def set_loss(self):
        """
        Instantiates the torch loss module based on self.config.loss
        :return:
        """
        loss_cls = eval(f"{self.config.loss_name}Loss")
        self.loss_fn = loss_cls(**self.config.loss_parameters)


    def set_optimizer(self):
        """
        Instantiates the torch optimizer based on self.config.optimizer
        :return:
        """
        trainable_parameters = [p for p in self.root.parameters() if p.requires_grad]
        optimizer_class = eval(f"torch.optim.{self.config.optimizer_name}")
        self.optimizer = optimizer_class(trainable_parameters,
                                         **self.config.optimizer_parameters)
        self.root.optimizer_group_id = 0

    def set_scheduler(self):
        """
        Instantiates the torch scheduler based on self.config.scheduler
        :return:
        """
        #TODO: option for no scheduler
        return
        scheduler_class = eval(f"torch.optim.{self.config.scheduler_name}")
        self.scheduler = scheduler_class(self.optimizer,
                                         **self.config.scheduler_parameters)

    def set_input_tensors_compatibility(self):
        """
        Push all the parameters on the appropriate device and cast them to the appropriate dtype
        :return:
        """
        self.to(self.config.input_tensors_device)
        self.type(self.input_space[self.wrapped_input_space_key].dtype)

    def set_tensorboard(self):
        if self.config.tb_folder is not None:
            if not os.path.exists(self.config.tb_folder):
                os.makedirs(self.config.tb_folder)
            self.logger = SummaryWriter(self.config.tb_folder)
        else:
            self.logger = None

        self.add_graph_to_tensorboard()


    def add_graph_to_tensorboard(self):
        # Save the graph in the logger
        if self.logger is not None:
            dummy_input = torch.Tensor(size=self.input_space[self.wrapped_input_space_key].shape).uniform_(0, 1)\
                .type(self.input_space[self.wrapped_input_space_key].dtype)\
                .to(self.config.input_tensors_device)\
                .unsqueeze(0)
            self.eval()
            with torch.no_grad():
                model_wrapper = ModelWrapper(self)
                self.logger.add_graph(model_wrapper, dummy_input, verbose=False)

    def forward(self, x):
        x = x.to(self.config.input_tensors_device)
        is_train = self.root.training
        if (x.shape[0] == 1) and is_train:
            self.root.eval()
            depth_first_traversal_outputs = self.root.depth_first_forward(x)
            self.root.train()
        else:
            depth_first_traversal_outputs = self.root.depth_first_forward(x)

        model_outputs = {}
        x_order_ids = []
        for leaf_idx in range(len(depth_first_traversal_outputs)):
            cur_node_path = depth_first_traversal_outputs[leaf_idx][0]
            cur_node_x_ids = depth_first_traversal_outputs[leaf_idx][1]
            cur_node_outputs = depth_first_traversal_outputs[leaf_idx][2]
            # stack results
            if not model_outputs:
                model_outputs["node_id"] = torch.Tensor([Node.path_to_id(path) for path in cur_node_path]).unsqueeze(-1)
                for k, v in cur_node_outputs.items():
                    model_outputs[k] = v
            else:
                model_outputs["node_id"] = torch.cat([model_outputs["node_id"], torch.Tensor([Node.path_to_id(path) for path in cur_node_path]).unsqueeze(-1)])
                for k, v in cur_node_outputs.items():
                    model_outputs[k] = torch.cat([model_outputs[k], v], dim=0)
            # save the sampled ids to reorder as in the input batch at the end
            x_order_ids += list(cur_node_x_ids)

        # reorder points
        sort_order = tuple(np.argsort(x_order_ids))
        for k, v in model_outputs.items():
                model_outputs[k] = v[sort_order, :]

        return model_outputs

    def forward_through_given_path(self, x, tree_desired_path):
        x = x.to(self.config.input_tensors_device)
        is_train = self.root.training
        if len(x) == 1 and is_train:
            self.root.eval()
            depth_first_traversal_outputs = self.root.depth_first_forward_through_given_path(x, tree_desired_path)
            self.root.train()
        else:
            depth_first_traversal_outputs = self.root.depth_first_forward_through_given_path(x, tree_desired_path)

        model_outputs = {}
        x_order_ids = []
        for leaf_idx in range(len(depth_first_traversal_outputs)):
            cur_node_path = depth_first_traversal_outputs[leaf_idx][0]
            cur_node_x_ids = depth_first_traversal_outputs[leaf_idx][1]
            cur_node_outputs = depth_first_traversal_outputs[leaf_idx][2]
            # stack results
            if not model_outputs:
                model_outputs["node_id"] = torch.Tensor([Node.path_to_id(path) for path in cur_node_path]).unsqueeze(-1)
                for k, v in cur_node_outputs.items():
                    model_outputs[k] = v
            else:
                model_outputs["node_id"] = torch.cat([model_outputs["node_id"], torch.Tensor([Node.path_to_id(path) for path in cur_node_path]).unsqueeze(-1)])
                for k, v in cur_node_outputs.items():
                    model_outputs[k] = torch.cat([model_outputs[k], v], dim=0)
            # save the sampled ids to reorder as in the input batch at the end
            x_order_ids += list(cur_node_x_ids)

        # reorder points
        sort_order = tuple(np.argsort(x_order_ids))
        for k, v in model_outputs.items():
                model_outputs[k] = v[sort_order, :]

        return model_outputs

    def calc_embedding(self, x, mode="niche", node_path=None, **kwargs):
        """
        :param mode: either "exhaustif" or "niche"
        :param node_path: if "niche" must specify niche's node_path
        :return:
        """
        x = x.to(self.config.input_tensors_device)\
            .type(self.input_space[self.wrapped_input_space_key].dtype)

        if node_path is None:
            z = {}
            for cur_node_path in self.root.get_node_pathes():
                z[cur_node_path] = torch.Tensor().new_full((len(x), self.config.encoder_n_latents), float("nan")) \
                    .to(self.config.input_tensors_device) \
                    .type(self.input_space[self.wrapped_input_space_key].dtype)
        else:
            z = torch.Tensor().new_full((len(x), self.config.encoder_n_latents), float("nan"))\
                    .to(self.config.input_tensors_device) \
                    .type(self.input_space[self.wrapped_input_space_key].dtype)\

        if mode == "niche":
            all_nodes_outputs = self.root.depth_first_forward_whole_branch_preorder(x)
        elif mode == "exhaustif":
            all_nodes_outputs = self.root.depth_first_forward_whole_tree_preorder(x)

        for node_idx in range(len(all_nodes_outputs)):
            cur_node_path = all_nodes_outputs[node_idx][0][0]

            if node_path is not None:
                if cur_node_path != node_path:
                    continue
                else:
                    cur_node_x_ids = all_nodes_outputs[node_idx][1]
                    cur_node_outputs = all_nodes_outputs[node_idx][2]
                    for idx in range(len(cur_node_x_ids)):
                        z[cur_node_x_ids[idx]] = cur_node_outputs["z"][idx]
                    break
            else:
                if mode == "niche":
                    cur_node_x_ids = all_nodes_outputs[node_idx][1]
                    cur_node_outputs = all_nodes_outputs[node_idx][2]
                    for idx in range(len(cur_node_x_ids)):
                        z[cur_node_path][cur_node_x_ids[idx]] = cur_node_outputs["z"][idx]
                elif mode == "exhaustif":
                    cur_node_outputs = all_nodes_outputs[node_idx][1]
                    z[cur_node_path] = cur_node_outputs["z"]
        return z


    def run_training(self, train_loader, n_epochs=0, valid_loader=None):

        do_validation = False
        if valid_loader is not None:
            best_valid_loss = sys.float_info.max
            do_validation = True

        for epoch in range(n_epochs):
            # check if elligible for split
            if self.config.split_active:
                self.trigger_split(train_loader)

            # perform train epoch
            t0 = time.time()
            train_losses = self.train_epoch(train_loader)
            t1 = time.time()

            if self.logger is not None and (self.n_epochs % self.config.tb_record_loss_frequency == 0):
                for k, v in train_losses.items():
                    self.logger.add_scalars('loss/{}'.format(k), {'train': v}, self.n_epochs)
                self.logger.add_text('time/train', 'Train Epoch {}: {:.3f} secs'.format(self.n_epochs, t1 - t0),
                                self.n_epochs)

            if do_validation:
                t2 = time.time()
                valid_losses = self.valid_epoch(valid_loader)
                t3 = time.time()
                if self.logger is not None and (self.n_epochs % self.config.tb_record_loss_frequency == 0):
                    for k, v in valid_losses.items():
                        self.logger.add_scalars('loss/{}'.format(k), {'valid': v}, self.n_epochs)
                    self.logger.add_text('time/valid', 'Valid Epoch {}: {:.3f} secs'.format(self.n_epochs, t3 - t2), self.n_epochs)

                #TODO: save_best_model if valid_loss['total] < best_loss


    def trigger_split(self, train_loader):

        splitted_leafs = []

        if (len(self.split_history) > self.config.n_max_splits) or \
                (self.n_epochs < self.config.n_min_epochs_before_split):
            return

        self.eval()
        train_fitness = None
        taken_pathes = []

        old_transform_state = train_loader.dataset.transform
        train_loader.dataset.transform = None

        with torch.no_grad():
            for data in train_loader:
                x = data["obs"].to(self.config.input_tensors_device)\
                    .type(self.input_space[self.wrapped_input_space_key].dtype)

                # forward
                model_outputs = self.forward(x)
                loss_inputs = {key: model_outputs[key] for key in self.loss_fn.input_keys_list}
                batch_losses = self.loss_fn(loss_inputs, reduction="none")
                cur_train_fitness = batch_losses[self.config.split_loss_key]

                # save losses
                if train_fitness is None:
                    train_fitness = np.expand_dims(cur_train_fitness.detach().cpu().numpy(), axis=-1)
                else:
                    train_fitness = np.vstack(
                        [train_fitness, np.expand_dims(cur_train_fitness.detach().cpu().numpy(), axis=-1)])
                # save taken pathes
                taken_pathes += [Node.id_to_path(int(node_id.item())) for node_id in model_outputs["node_id"]]

        for leaf_path in list(set(taken_pathes)):
            leaf_node = self.root.get_child_node(leaf_path)
            leaf_x_ids = (np.array(taken_pathes, copy=False) == leaf_path)
            split_x_fitness = train_fitness[leaf_x_ids, :]
            leaf_node.fitness_last_epochs.append(split_x_fitness.mean())

            if leaf_path == "0":
                n_epochs_since_split = self.n_epochs
            else:
                n_epochs_since_split = (self.n_epochs - self.split_history[leaf_path[:-1]]["epoch"])
            if (n_epochs_since_split < self.config.n_min_epochs_between_splits) or \
                    (len(leaf_x_ids) < self.config.n_min_points_for_split):
                continue

            trigger_split_in_leaf = False
            if self.config.split_type == "threshold":
                poor_buffer = (split_x_fitness > self.config.split_parameters_threshold).squeeze()
                if (poor_buffer.sum() > self.config.split_parameters_n_max_bad_points):
                    trigger_split_in_leaf = True

            elif self.config.split_type == "plateau":
                if len(leaf_node.fitness_last_epochs) > self.config.split_parameters_n_steps_average:
                    leaf_node.fitness_last_epochs.pop(0)
                fitness_vals = np.asarray(leaf_node.fitness_last_epochs)
                fitness_speed_last_epochs = fitness_vals[1:] - fitness_vals[:-1]
                running_average_speed = np.abs(fitness_speed_last_epochs.mean())
                if (running_average_speed < self.config.split_parameters_epsilon):
                    trigger_split_in_leaf = True

            if trigger_split_in_leaf:
                self.split(leaf_path, create_boundary=True)
                # Save the new graph in the logger
                self.add_graph_to_tensorboard()

                splitted_leafs.append(leaf_path)

        train_loader.dataset.transform = old_transform_state
        return splitted_leafs

    def split(self, leaf_path, create_boundary=False):
            leaf_node = self.root.get_child_node(leaf_path)
            leaf_node.split(create_boundary=create_boundary, history=self._access_history()['output'], boundary_name=self.config.boundary_name, boundary_parameters=self.config.boundary_parameters)

            # 1) set input tensors compatibility
            self.set_input_tensors_compatibility()

            # 2) update optimizer
            cur_node_optimizer_group_id = leaf_node.optimizer_group_id
            # delete param groups and residuates in optimize.state
            del self.optimizer.param_groups[cur_node_optimizer_group_id]
            for n, p in leaf_node.named_parameters():
                if p in self.optimizer.state.keys():
                    del self.optimizer.state[p]
            leaf_node.optimizer_group_id = None
            n_groups = len(self.optimizer.param_groups)
            # update optimizer_group ids in the tree and sanity check that there is no conflict
            sanity_check = np.asarray([False] * n_groups)
            for other_leaf_path in self.root.get_leaf_pathes():
                if other_leaf_path[:len(leaf_path)] != leaf_path:
                    other_leaf = self.root.get_child_node(other_leaf_path)
                    if other_leaf.optimizer_group_id > cur_node_optimizer_group_id:
                        other_leaf.optimizer_group_id -= 1
                    if sanity_check[other_leaf.optimizer_group_id] == False:
                        sanity_check[other_leaf.optimizer_group_id] = True
                    else:
                        raise ValueError("doublons in the optimizer group ids")
            if (n_groups > 0) and (~sanity_check).any():
                raise ValueError("optimizer group ids does not match the optimzer param groups length")
            self.optimizer.add_param_group({"params": leaf_node.left.parameters()})
            leaf_node.left.optimizer_group_id = n_groups
            self.optimizer.add_param_group({"params": leaf_node.right.parameters()})
            leaf_node.right.optimizer_group_id = n_groups + 1


            # 3) Create new keys in output space
            self.output_space.spaces[f"{leaf_path}0"] = BoxSpace(low=0, high=0, shape=(ConfigParameterBinding("encoder_n_latents"),))
            self.output_space.spaces[f"{leaf_path}0"].initialize(self)
            self.output_space.spaces[f"{leaf_path}1"] = BoxSpace(low=0, high=0, shape=(ConfigParameterBinding("encoder_n_latents"),))
            self.output_space.spaces[f"{leaf_path}1"].initialize(self)

            # 4) Update history
            self._call_output_history_update()

            # 5) Expand new goal spaces
            #TODO

            # save split history
            self.split_history[leaf_path] = {"boundary": leaf_node.boundary, "epoch": self.n_epochs}


    def train_epoch(self, train_loader):
        if (len(self.split_history) > 0) and self.config.alternated_backward_active:
            if (self.n_epochs % self.config.alternated_backward_period) < self.config.alternated_backward_connections:
                # train only connections
                for leaf_path in self.root.get_leaf_pathes():
                    leaf_node = self.root.get_child_node(leaf_path)
                    for n, p in leaf_node.network.named_parameters():
                        if "_c" not in n:
                            p.requires_grad = False
                        else:
                            p.requires_grad = True
            else:
                # train only children module without connections
                for leaf_path in self.root.get_leaf_pathes():
                    leaf_node = self.root.get_child_node(leaf_path)
                    for n, p in leaf_node.network.named_parameters():
                        if "_c" not in n:
                            p.requires_grad = True
                        else:
                            p.requires_grad = False


        self.train()

        taken_pathes = []
        losses = {}

        with torch.set_grad_enabled(True):
            for data in train_loader:
                x = data['obs'].to(self.config.input_tensors_device)\
                                .type(self.input_space[self.wrapped_input_space_key].dtype)
                #TODO: see why the dtype is modified through TinyDB?
                # forward
                model_outputs = self.forward(x)
                loss_inputs = {key: model_outputs[key] for key in self.loss_fn.input_keys_list}
                batch_losses = self.loss_fn(loss_inputs, reduction="none")
                # backward
                loss = batch_losses['total'].mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # save losses
                for k, v in batch_losses.items():
                    if k not in losses:
                        losses[k] = v.detach().cpu()
                    else:
                        losses[k] = torch.cat([losses[k], v.detach().cpu()])
                # save taken path
                taken_pathes += [Node.id_to_path(int(node_id.item())) for node_id in model_outputs["node_id"]]

        # Logger save results per leaf
        if self.logger is not None:
            for leaf_path in list(set(taken_pathes)):
                if len(leaf_path) > 1:
                    leaf_x_ids = np.where(np.array(taken_pathes, copy=False) == leaf_path)[0]
                    for k, v in losses.items():
                        leaf_v = v[leaf_x_ids]
                        self.logger.add_scalars('loss/{}'.format(k), {'train-{}'.format(leaf_path): leaf_v.mean()},
                                           self.n_epochs)

        for k, v in losses.items():
            losses[k] = v.mean()

        self.n_epochs += 1

        return losses

    def valid_epoch(self, valid_loader):
        self.eval()

        taken_pathes = []
        losses = {}

        # Prepare logging
        record_valid_images = False
        record_embeddings = False
        if self.logger is not None:
            if self.n_epochs % self.config.tb_record_images_frequency == 0:
                record_valid_images = True
                images = torch.empty((0, ) + self.input_space[self.wrapped_input_space_key].shape)
                recon_images = torch.empty((0, ) + self.input_space[self.wrapped_input_space_key].shape)
            if self.n_epochs % self.config.tb_record_embeddings_frequency == 0:
                record_embeddings = True
                embeddings = torch.empty((0, ) + self.output_space["0"].shape)
                labels = torch.empty((0, 1))
                if images is None:
                    images = torch.empty((0, ) + self.input_space[self.wrapped_input_space_key].shape)

        with torch.no_grad():
            for data in valid_loader:
                x = data['obs'].to(self.config.input_tensors_device)\
                    .type(self.input_space[self.wrapped_input_space_key].dtype)
                #TODO: see why the dtype is modified through TinyDB?
                # forward
                model_outputs = self.forward(x)
                loss_inputs = {key: model_outputs[key] for key in self.loss_fn.input_keys_list}
                batch_losses = self.loss_fn(loss_inputs, reduction="none")
                # save losses
                for k, v in batch_losses.items():
                    if k not in losses:
                        losses[k] = v.detach().cpu()
                    else:
                        losses[k] = torch.cat([losses[k], v.detach().cpu()])

                if record_valid_images:
                    recon_x = model_outputs["recon_x"].cpu().detach()
                    if len(images) < self.config.tb_record_memory_max:
                        images = torch.cat([images, x.cpu().detach()])
                    if len(recon_images) < self.config.tb_record_memory_max:
                        recon_images = torch.cat([recon_images, recon_x])

                if record_embeddings:
                    if len(embeddings) < self.config.tb_record_memory_max:
                        embeddings = torch.cat([embeddings, model_outputs["z"].cpu().detach().view(x.shape[0], self.config.encoder_n_latents)])
                        labels = torch.cat([labels, data["label"].cpu()])
                    if not record_valid_images:
                        if len(images) < self.config.tb_record_memory_max:
                            images = torch.cat([images, x.cpu().detach()])

                taken_pathes += [Node.id_to_path(int(node_id.item())) for node_id in model_outputs["node_id"]]


        # LOGGER SAVE RESULT PER LEAF
        if self.logger is not None:
            for leaf_path in list(set(taken_pathes)):
                leaf_x_ids = np.where(np.array(taken_pathes, copy=False) == leaf_path)[0]

                if len(leaf_path) > 1:
                    for k, v in losses.items():
                        leaf_v = v[leaf_x_ids]
                        self.logger.add_scalars('loss/{}'.format(k), {'valid-{}'.format(leaf_path): leaf_v.mean()}, self.n_epochs)

                if record_embeddings:
                    leaf_embeddings = embeddings[leaf_x_ids]
                    leaf_labels = labels[leaf_x_ids]
                    leaf_images = images[leaf_x_ids]
                    if len(leaf_images.shape) == 5:
                        leaf_images = leaf_images[:, :, leaf_images.shape[2] // 2, :, :]  # we take slice at middle depth only
                    if (leaf_images.shape[1] != 1) or (leaf_images.shape[1] != 3):
                        leaf_images = leaf_images[:, :3, ...]
                    leaf_images = resize_embeddings(leaf_images)
                    try:
                        self.logger.add_embedding(
                            leaf_embeddings,
                            metadata=leaf_labels,
                            label_img=leaf_images,
                            global_step=self.n_epochs,
                            tag="leaf_{}".format(leaf_path))
                    except:
                        pass

                if record_valid_images:
                    n_images = min(len(leaf_x_ids), 40)
                    sampled_ids = np.random.choice(len(leaf_x_ids), n_images, replace=False)
                    input_images = images[leaf_x_ids[sampled_ids]].cpu()
                    output_images = recon_images[leaf_x_ids[sampled_ids]].cpu()
                    if self.config.loss.parameters.reconstruction_dist == "bernoulli":
                        output_images = torch.sigmoid(output_images)
                    vizu_tensor_list = [None] * (2 * n_images)
                    vizu_tensor_list[0::2] = [input_images[n] for n in range(n_images)]
                    vizu_tensor_list[1::2] = [output_images[n] for n in range(n_images)]
                    logger_add_image_list(self.logger, vizu_tensor_list, f"leaf_{leaf_path}/reconstructions",
                                          global_step=self.n_epochs,
                                          n_channels=self.input_space[self.wrapped_input_space_key].shape[0],
                                          spatial_dims=len(self.input_space[self.wrapped_input_space_key].shape[1:]))

        # 4) AVERAGE LOSS ON WHOLE TREE AND RETURN
        for k, v in losses.items():
            losses[k] = v.mean().item()

        return losses

    def train_update(self):
        train_dataset = ExperimentHistoryDataset(access_history_fn=self._access_history,
                                                 key=f"input.{self.wrapped_input_space_key}",
                                                 history_ids=list(range(self.CURRENT_RUN_INDEX)),
                                                 filter=lambda x: ((x - x[0]) < 1e-10).all().item())
        train_loader = DataLoader(train_dataset,
                                  # TODO: sampler=weighted_train_sampler,
                                  batch_size=self.config.dataloader_batch_size,
                                  num_workers=self.config.dataloader_num_workers,
                                  drop_last=self.config.dataloader_drop_last,
                                  # TODO: collate_fn=self.config.dataloader_collate_fn
                                  )

        valid_dataset = ExperimentHistoryDataset(access_history_fn=self._access_history,
                                                 key=f"input.{self.wrapped_input_space_key}",
                                                 history_ids=list(range(-min(20, self.CURRENT_RUN_INDEX), 0)),
                                                 filter=lambda x: ((x - x[0]) < 1e-10).all().item())

        valid_loader = DataLoader(valid_dataset,
                                  batch_size=self.config.dataloader_batch_size,
                                  num_workers=self.config.dataloader_num_workers,
                                  drop_last=False,
                                  # TODO: collate_fn=self.config.dataloader_collate_fn
                                  )

        self.run_training(train_loader=train_loader, n_epochs=self.config.n_epochs_per_train_period, valid_loader=valid_loader)

        self._call_output_history_update()


    def map(self, observations, is_output_new_discovery, **kwargs):

        if (self.CURRENT_RUN_INDEX % self.config.train_period == 0) and (self.CURRENT_RUN_INDEX > 0) and is_output_new_discovery:
            self.train_update()

        input = observations[self.wrapped_input_space_key] \
            .to(self.config.input_tensors_device) \
            .type(self.input_space[self.wrapped_input_space_key].dtype) \
            .unsqueeze(0)

        output = self.calc_embedding(input)
        output = map_nested_dicts(output, lambda x: x.flatten().cpu().detach())

        if self.config.expand_output_space:
            self.output_space.expand(output)

        return output

    def save(self):
        return {
            "epoch": self.n_epochs,
            "root_state_dict": self.root.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "split_history": self.split_history
        }

    def load(self, saved_dict):
        #TODO: deal with device
        self.n_epochs = saved_dict["epoch"]
        self.split_history = saved_dict['split_history']
        for split_node_path, split_node_attr in self.split_history.items():
            node_to_split = self.root.get_child_node(split_node_path)
            self.split(split_node_path, create_boundary=False)
            node_to_split.boundary = split_node_attr["boundary"]

        self.root.load_state_dict(saved_dict['root_state_dict'])
        self.optimizer.load_state_dict(saved_dict['optimizer_state_dict'])
