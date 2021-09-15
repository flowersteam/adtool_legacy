from auto_disc.output_representations import BaseOutputRepresentation
from auto_disc.utils.config_parameters import IntegerConfigParameter, StringConfigParameter, BooleanConfigParameter
from auto_disc.utils.spaces.utils import distance, ConfigParameterBinding
from auto_disc.utils.spaces import DictSpace, BoxSpace
import math
import numpy as np
import os
import sys
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter


@StringConfigParameter(name="input_tensors_layout", default="dense", possible_values=["dense", "minkowski", ])
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

@StringConfigParameter(name="optimizer_name", default="Adam", possible_values=["Adam", ])
#TODO: DictConfigParameter for optimizer_parameters

@BooleanConfigParameter(name="checkpoint_active", default=True)
# TODO: replace checkpoint frequency with callbacks
@StringConfigParameter(name="checkpoint_folder", default="./checkpoints", possible_values="all")
@BooleanConfigParameter(name="checkpoint_best_model", default=False)
@IntegerConfigParameter(name="checkpoint_frequency", default=10, min=1)

@BooleanConfigParameter(name="tensorboard_active", default=True)
# TODO: replace tensorboard frequency with callbacks
@StringConfigParameter(name="tb_folder", default="./tensorboard", possible_values="all")
@IntegerConfigParameter(name="tb_record_loss_frequency", default=1, min=1)
@IntegerConfigParameter(name="tb_record_images_frequency", default=10, min=1)
@IntegerConfigParameter(name="tb_record_embeddings_frequency", default=10, min=1)
@IntegerConfigParameter(name="tb_record_memory_max", default=100, min=1)

@IntegerConfigParameter(name="train_period", default=10, min=1)

class VAE(nn.Module, BaseOutputRepresentation):
    """
    Custom VAE Output Representation
    """

    output_space = DictSpace(
        embedding=BoxSpace(low=0, high=0, shape=(ConfigParameterBinding("encoder_n_latents"),))
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

        # Checkpoints
        self.set_checkpoint()

        # Tensorboard Logger
        self.set_tensorboard()

        # set counter
        self.n_epochs = 0


    def set_model(self):
        """
        Instantiates the torch model weights based on self.config.encoder
        The model must have an encoder and define the input shape and output shape (of that encoder)
        :return:
        """
        encoder_cls = eval(f"{self.config.encoder_name}Encoder")
        self.encoder = encoder_cls(input_shape=self.input_space[self.wrapped_input_space_key].shape,
                                   n_latents=self.config.encoder_n_latents,
                                   n_conv_layers=self.config.encoder_n_conv_layers,
                                   feature_layer=self.config.encoder_feature_layer,
                                   hidden_channels=self.config.encoder_hidden_channels,
                                   hidden_dims=self.config.encoder_hidden_dims,
                                   conditional_type=self.config.encoder_conditional_type,
                                   use_attention=self.config.encoder_use_attention,
                                   )
        decoder_cls = eval(f"{self.config.encoder_name}Decoder")
        self.decoder = decoder_cls(output_shape=self.input_space[self.wrapped_input_space_key].shape,
                                   n_latents=self.config.encoder_n_latents,
                                   n_conv_layers=self.config.encoder_n_conv_layers,
                                   feature_layer=self.config.encoder_feature_layer,
                                   hidden_channels=self.config.encoder_hidden_channels,
                                   hidden_dims=self.config.encoder_hidden_dims,
                                   )
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
        optimizer_class = eval(f"torch.optim.{self.config.optimizer_name}")
        self.optimizer = optimizer_class(self.parameters(),
                                         **self.config.optimizer_parameters)

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

        # Save the graph in the logger
        if self.logger is not None:
            dummy_input = torch.Tensor(size=self.input_space[self.wrapped_input_space_key].shape).uniform_(0, 1)\
                .type(self.input_space[self.wrapped_input_space_key].dtype)\
                .to(self.config.input_tensors_device)\
                .unsqueeze(0)
            self.eval()
            with torch.no_grad():
                self.logger.add_graph(self, dummy_input, verbose=False)

    def set_checkpoint(self):
        if self.config.checkpoint_folder is not None:
            if not os.path.exists(self.config.checkpoint_folder):
                os.makedirs(self.config.checkpoint_folder)


    def forward_from_encoder(self, encoder_outputs):
        decoder_outputs = self.decoder(encoder_outputs["z"])
        model_outputs = encoder_outputs
        model_outputs.update(decoder_outputs)
        return model_outputs

    def forward(self, x):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x)

        x = x.to(self.config.input_tensors_device)
        encoder_outputs = self.encoder(x)
        return self.forward_from_encoder(encoder_outputs)

    def forward_for_graph_tracing(self, x):
        x = x.to(self.config.input_tensors_device)
        z, feature_map = self.encoder.forward_for_graph_tracing(x)
        recon_x = self.decoder(z)
        return recon_x

    def run_training(self, train_loader, n_epochs=0, valid_loader=None):

        do_validation = False
        if valid_loader is not None:
            best_valid_loss = sys.float_info.max
            do_validation = True

        for epoch in range(n_epochs):
            t0 = time.time()
            train_losses = self.train_epoch(train_loader)
            t1 = time.time()

            if self.logger is not None and (self.n_epochs % self.config.tb_record_loss_frequency == 0):
                for k, v in train_losses.items():
                    self.logger.add_scalars('loss/{}'.format(k), {'train': v}, self.n_epochs)
                self.logger.add_text('time/train', 'Train Epoch {}: {:.3f} secs'.format(self.n_epochs, t1 - t0),
                                self.n_epochs)

            if self.n_epochs % self.config.checkpoint_frequency == 0:
                #TODO: proper save function with AutoDiscTool, torch.save throws error here
                #torch.save(self, os.path.join(self.config.checkpoint_folder, 'current_weight_model.pth'))
                pass

            if do_validation:
                t2 = time.time()
                valid_losses = self.valid_epoch(valid_loader)
                t3 = time.time()
                if self.logger is not None and (self.n_epochs % self.config.tb_record_loss_frequency == 0):
                    for k, v in valid_losses.items():
                        self.logger.add_scalars('loss/{}'.format(k), {'valid': v}, self.n_epochs)
                    self.logger.add_text('time/valid', 'Valid Epoch {}: {:.3f} secs'.format(self.n_epochs, t3 - t2),
                                    self.n_epochs)

                valid_loss = valid_losses['total']
                if valid_loss < best_valid_loss and self.config.checkpoint_best_model:
                    best_valid_loss = valid_loss
                    self.save(os.path.join(self.config.checkpoint_folderolder, 'best_weight_model.pth'))

    def train_epoch(self, train_loader):
        self.train()
        losses = {}
        with torch.set_grad_enabled(True):
            for data in train_loader:
                x = data['obs'].to(self.config.input_tensors_device)\
                                .type(self.input_space[self.wrapped_input_space_key].dtype)
                #TODO: see why the dtype is modified through TinyDB?
                # forward
                model_outputs = self.forward(x)
                loss_inputs = {key: model_outputs[key] for key in self.loss_fn.input_keys_list}
                batch_losses = self.loss_fn(loss_inputs)
                # backward
                loss = batch_losses['total']
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # save losses
                for k, v in batch_losses.items():
                    if k not in losses:
                        losses[k] = [v.data.item()]
                    else:
                        losses[k].append(v.data.item())

        for k, v in losses.items():
            losses[k] = torch.mean(torch.tensor(v))

        self.n_epochs += 1

        return losses

    def valid_epoch(self, valid_loader):
        self.eval()
        losses = {}

        # Prepare logging
        record_valid_images = False
        record_embeddings = False
        if self.logger is not None:
            if self.n_epochs % self.config.tb_record_images_frequency == 0:
                record_valid_images = True
                images = []
                recon_images = []
            if self.n_epochs % self.config.tb_record_embeddings_frequency == 0:
                record_embeddings = True
                embeddings = []
                labels = []
                if images is None:
                    images = []
        with torch.no_grad():
            for data in valid_loader:
                x = data['obs'].to(self.config.input_tensors_device)\
                    .type(self.input_space[self.wrapped_input_space_key].dtype)
                #TODO: see why the dtype is modified through TinyDB?
                # forward
                model_outputs = self.forward(x)
                loss_inputs = {key: model_outputs[key] for key in self.loss_fn.input_keys_list}
                batch_losses = self.loss_fn(loss_inputs)
                # save losses
                for k, v in batch_losses.items():
                    if k not in losses:
                        losses[k] = v.detach().cpu().unsqueeze(-1)
                    else:
                        losses[k] = torch.vstack([losses[k], v.detach().cpu().unsqueeze(-1)])

                if record_valid_images:
                    recon_x = model_outputs["recon_x"].cpu().detach()
                    if len(images) < self.config.tb_record_memory_max:
                        images.append(x.cpu().detach())
                    if len(recon_images) < self.config.tb_record_memory_max:
                        recon_images.append(recon_x)

                if record_embeddings:
                    if len(embeddings) < self.config.tb_record_memory_max:
                        embeddings.append(model_outputs["z"].cpu().detach().view(x.shape[0], self.config.encoder_n_latents))
                        labels.append(data["label"])
                    if not record_valid_images:
                        if len(images) < self.config.tb_record_memory_max:
                            images.append(x.cpu().detach())

        if record_valid_images:
            recon_images = torch.cat(recon_images)
            images = torch.cat(images)
        if record_embeddings:
            embeddings = torch.cat(embeddings)
            labels = torch.cat(labels)
            if not record_valid_images:
                images = torch.cat(images)

        # log results
        if record_valid_images:
            n_images = min(len(images), 40)
            sampled_ids = torch.randperm(len(images))[:n_images]
            input_images = images[sampled_ids].detach().cpu()
            output_images = recon_images[sampled_ids].detach().cpu()
            if self.loss_fn.reconstruction_dist == "bernoulli":
                output_images = torch.sigmoid(output_images)
            vizu_tensor_list = [None] * (2 * n_images)
            vizu_tensor_list[0::2] = [input_images[n] for n in range(n_images)]
            vizu_tensor_list[1::2] = [output_images[n] for n in range(n_images)]
            logger_add_image_list(self.logger, vizu_tensor_list, "reconstructions",
                                  global_step=self.n_epochs, n_channels=self.input_space[self.wrapped_input_space_key].shape[0],
                                  spatial_dims=len(self.input_space[self.wrapped_input_space_key].shape[1:]))


        if record_embeddings:
            if len(images.shape) == 5:
                images = images[:, :, self.input_space[self.wrapped_input_space_key].shape[0] // 2, :, :] #we take slice at middle depth only
            if (images.shape[1] != 1) or (images.shape[1] != 3):
                images = images[:, :3, ...]
            images = resize_embeddings(images)
            self.logger.add_embedding(
                embeddings,
                metadata=labels,
                label_img=images,
                global_step=self.n_epochs)

        # average loss and return
        for k, v in losses.items():
            losses[k] = torch.mean(torch.tensor(v)).item()

        return losses

    def train_update(self):
        train_dataset = ExperimentHistoryDataset(access_history_fn=self._access_history,
                                                 history_ids=list(range(self.CURRENT_RUN_INDEX)),
                                                 wrapped_input_space_key=self.wrapped_input_space_key)
        train_loader = DataLoader(train_dataset)
        valid_dataset = ExperimentHistoryDataset(access_history_fn=self._access_history,
                                                 history_ids=list(range(-5,0)),
                                                 wrapped_input_space_key=self.wrapped_input_space_key)
        valid_loader = DataLoader(valid_dataset)
        self.run_training(train_loader=train_loader, n_epochs=10, valid_loader=valid_loader)


    def map(self, observations, is_output_new_discovery, **kwargs):

        if (self.CURRENT_RUN_INDEX % self.config.train_period == 0) and (self.CURRENT_RUN_INDEX > 0) and is_output_new_discovery:
            self.train_update()

        input = observations[self.wrapped_input_space_key] \
            .to(self.config.input_tensors_device) \
            .type(self.input_space[self.wrapped_input_space_key].dtype) \
            .unsqueeze(0)
        output = self.encoder.calc_embedding(input).flatten().cpu().detach()

        return {"embedding": output}

    def calc_distance(self, embedding_a, embedding_b, **kwargs):
        # L2 + add regularizer to avoid dead outcomes
        dist = distance.calc_l2(embedding_a, embedding_b)
        return dist


"""==============================================================================================================
ENCODERS
==============================================================================================================="""


class Encoder(nn.Module):
    """
    Base Encoder class
    User can specify variable number of convolutional layers to deal with images of different sizes (eg: 3 layers for 32*32, 6 layers for 256*256 images)

    Parameters
    ----------
    n_channels: number of channels
    input_size: size of input image
    n_conv_layers: desired number of convolutional layers
    n_latents: dimensionality of the infered latent output.
    """

    def __init__(self, input_shape=(1,64,64),
                        n_latents=10,
                        n_conv_layers=4,
                        feature_layer=1,
                        hidden_channels=32,
                        hidden_dims=256,
                        conditional_type="gaussian",
                        use_attention=False):
        nn.Module.__init__(self)
        self.input_shape = input_shape
        self.n_latents = n_latents
        self.n_conv_layers = n_conv_layers
        self.feature_layer = feature_layer
        self.hidden_dims = hidden_dims
        self.hidden_channels = hidden_channels
        self.conditional_type = conditional_type
        self.use_attention = use_attention

        
        self.spatial_dims = len(self.input_shape) - 1
        assert 2 <= self.spatial_dims <= 3, "Image must be 2D or 3D"
        if self.spatial_dims == 2:
            self.conv_module = nn.Conv2d
            self.maxpool_module = nn.MaxPool2d
            self.batchnorm_module = nn.BatchNorm2d
        elif self.spatial_dims == 3:
            self.conv_module = nn.Conv3d
            self.maxpool_module = nn.MaxPool3d
            self.batchnorm_module = nn.BatchNorm3d

        self.output_keys_list = ["x", "lf", "gf", "z"]
        if self.conditional_type == "gaussian":
            self.output_keys_list += ["mu", "logvar"]

    def forward(self, x):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x)

        # local feature map
        lf = self.lf(x)
        # global feature map
        gf = self.gf(lf)

        encoder_outputs = {"x": x, "lf": lf, "gf": gf}

        # encoding
        if self.conditional_type == "gaussian":
            mu, logvar = torch.chunk(self.ef(gf), 2, dim=1)
            z = self.reparameterize(mu, logvar)
            if z.ndim > 2:
                for _ in range(self.spatial_dims):
                    mu = mu.squeeze(dim=-1)
                    logvar = logvar.squeeze(dim=-1)
                    z = z.squeeze(dim=-1)
            encoder_outputs.update({"z": z, "mu": mu, "logvar": logvar})
        elif self.conditional_type == "deterministic":
            z = self.ef(gf)
            if z.ndim > 2:
                for _ in range(self.spatial_dims):
                    z = z.squeeze(dim=-1)
            encoder_outputs.update({"z": z})

        # attention features
        if self.use_attention:
            af = self.af(gf)
            af = F.normalize(af, p=2)
            encoder_outputs.update({"af": af})

        return encoder_outputs

    def forward_for_graph_tracing(self, x):
        lf = self.lf(x)
        gf = self.gf(lf)
        if self.conditional_type == "gaussian":
            mu, logvar = torch.chunk(self.ef(gf), 2, dim=1)
            z = self.reparameterize(mu, logvar)
        else:
            z = self.ef(gf)
        return z, lf

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def calc_embedding(self, x):
        encoder_outputs = self.forward(x)
        return encoder_outputs["z"]


def get_encoder(model_architecture):
    """
    model_architecture: string such that the class encoder called is <model_architecture>Encoder
    """
    return eval("{}Encoder".format(model_architecture))


class BurgessEncoder(Encoder):
    """ 
    Extended Encoder of the model proposed in Burgess et al. "Understanding disentangling in $\beta$-VAE"

    Model Architecture (transposed for decoder)
    ------------
    - Convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
    - 2 fully connected layers (each of 256 units)
    - Latent distribution:
        - 1 fully connected layer of 2*n_latents units (log variance and mean for Gaussians distributions)
    """

    def __init__(self, input_shape=(1,64,64), n_latents=10, n_conv_layers=4, feature_layer=1, hidden_channels=32, hidden_dims=256, conditional_type="gaussian", use_attention=False):
        Encoder.__init__(self, input_shape=input_shape, n_latents=n_latents, n_conv_layers=n_conv_layers, feature_layer=feature_layer, hidden_channels=hidden_channels, hidden_dims=hidden_dims, conditional_type=conditional_type, use_attention=use_attention)

        # need square image input
        assert torch.all(torch.tensor([self.input_shape[i] == self.input_shape[1] for i in range(2, len(self.input_shape))])), "BurgessEncoder needs a square image input size"

        # network architecture
        kernels_size = [4] * self.n_conv_layers
        strides = [2] * self.n_conv_layers
        pads = [1] * self.n_conv_layers
        dils = [1] * self.n_conv_layers

        # feature map size
        feature_map_sizes = conv_output_sizes(self.input_shape[1:], self.n_conv_layers, kernels_size, strides,
                                              pads, dils)

        # local feature
        self.local_feature_shape = (
            self.hidden_channels, feature_map_sizes[self.feature_layer][0],
            feature_map_sizes[self.feature_layer][1])
        self.lf = nn.Sequential()
        for conv_layer_id in range(self.feature_layer + 1):
            if conv_layer_id == 0:
                self.lf.add_module("conv_{}".format(0), nn.Sequential(
                    self.conv_module(self.input_shape[0], self.hidden_channels, kernels_size[0], strides[0], pads[0],
                                     dils[0]),
                    nn.ReLU()))
            else:
                self.lf.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                    self.conv_module(self.hidden_channels, self.hidden_channels, kernels_size[conv_layer_id],
                                     strides[conv_layer_id],
                                     pads[conv_layer_id], dils[conv_layer_id]), nn.ReLU()))
        self.lf.out_connection_type = ("conv", self.hidden_channels)

        # global feature
        self.gf = nn.Sequential()
        ## convolutional layers
        for conv_layer_id in range(self.feature_layer + 1, self.n_conv_layers):
            self.gf.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                self.conv_module(self.hidden_channels, self.hidden_channels, kernels_size[conv_layer_id], strides[conv_layer_id],
                                 pads[conv_layer_id], dils[conv_layer_id]), nn.ReLU()))
        self.gf.add_module("flatten", Flatten())
        ## linear layers
        n_linear_in = self.hidden_channels * torch.prod(torch.tensor(feature_map_sizes[-1])).item()
        self.gf.add_module("lin_0",
                           nn.Sequential(nn.Linear(n_linear_in, self.hidden_dims),
                                         nn.ReLU()))
        self.gf.add_module("lin_1", nn.Sequential(nn.Linear(self.hidden_dims, self.hidden_dims), nn.ReLU()))
        self.gf.out_connection_type = ("lin", self.hidden_dims)

        # encoding feature
        if self.conditional_type == "gaussian":
            self.add_module("ef", nn.Linear(self.hidden_dims, 2 * self.n_latents))
        elif self.conditional_type == "deterministic":
            self.add_module("ef", nn.Linear(self.hidden_dims, self.n_latents))
        else:
            raise ValueError("The conditional type must be either gaussian or deterministic")

        # attention feature
        if self.use_attention:
            self.add_module("af", nn.Linear(self.hidden_dims, 4 * self.n_latents))


class HjelmEncoder(Encoder):
    """ 
    Extended Encoder of the model proposed in Hjelm et al. "Learning deep representations by mutual information estimation and maximization"

    Model Architecture (transposed for decoder)
    ------------
    - Convolutional layers (64-138-256-512 channels for 64*64 image), (4 x 4 kernel), (stride of 2) => for a MxM feature map
    - 1 fully connected layers (1024 units)
    - Latent distribution:
        - 1 fully connected layer of n_latents units
    """

    def __init__(self, input_shape=(1, 64, 64), n_latents=10, n_conv_layers=4, feature_layer=1, hidden_channels=32, hidden_dims=256,
                 conditional_type="gaussian", use_attention=False):
        Encoder.__init__(self, input_shape=input_shape, n_latents=n_latents, n_conv_layers=n_conv_layers, feature_layer=feature_layer,
                         hidden_channels=hidden_channels, hidden_dims=hidden_dims, conditional_type=conditional_type,
                         use_attention=use_attention)

        # need square image input
        assert torch.all(torch.tensor([self.input_shape[i] == self.input_shape[1] for i in range(2, len(self.input_shape))])), "HjelmEncoder needs a square image input size"

        # network architecture
        kernels_size = [4] * self.n_conv_layers
        strides = [2] * self.n_conv_layers
        pads = [1] * self.n_conv_layers
        dils = [1] * self.n_conv_layers

        # feature map size
        feature_map_sizes = conv_output_sizes(self.input_shape[1:], self.n_conv_layers, kernels_size,
                                              strides, pads, dils)

        # local feature 
        ## convolutional layers
        self.local_feature_shape = (
            int(self.hidden_channels * math.pow(2, self.feature_layer)),
            feature_map_sizes[self.feature_layer][0],
            feature_map_sizes[self.feature_layer][1])
        self.lf = nn.Sequential()
        for conv_layer_id in range(self.feature_layer + 1):
            if conv_layer_id == 0:
                self.lf.add_module("conv_{}".format(0), nn.Sequential(
                    self.conv_module(self.input_shape[0], self.hidden_channels, kernels_size[0], strides[0], pads[0],
                                     dils[0]),
                    nn.ReLU()))
            else:
                self.lf.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                    self.conv_module(self.hidden_channels, self.hidden_channels * 2, kernels_size[conv_layer_id],
                                     strides[conv_layer_id],
                                     pads[conv_layer_id], dils[conv_layer_id]),
                    self.batchnorm_module(self.hidden_channels * 2),
                    nn.ReLU()))
                self.hidden_channels *= 2
        self.lf.out_connection_type = ("conv", self.hidden_channels)

        # global feature 
        self.gf = nn.Sequential()
        ## convolutional layers
        for conv_layer_id in range(self.feature_layer + 1, self.n_conv_layers):
            self.gf.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                self.conv_module(self.hidden_channels, self.hidden_channels * 2, kernels_size[conv_layer_id],
                                 strides[conv_layer_id],
                                 pads[conv_layer_id], dils[conv_layer_id]), self.batchnorm_module(self.hidden_channels * 2),
                nn.ReLU()))
            self.hidden_channels *= 2
        self.gf.add_module("flatten", Flatten())
        ## linear layers
        n_linear_in = self.hidden_channels * torch.prod(torch.tensor(feature_map_sizes[-1])).item()
        self.gf.add_module("lin_0",
                           nn.Sequential(nn.Linear(n_linear_in, self.hidden_dims),
                                         nn.BatchNorm1d(self.hidden_dims), nn.ReLU()))
        self.gf.out_connection_type = ("lin", self.hidden_dims)

        # encoding feature
        if self.conditional_type == "gaussian":
            self.add_module("ef", nn.Linear(self.hidden_dims, 2 * self.n_latents))
        elif self.conditional_type == "deterministic":
            self.add_module("ef", nn.Linear(self.hidden_dims, self.n_latents))
        else:
            raise ValueError("The conditional type must be either gaussian or deterministic")

        # attention feature
        if self.use_attention:
            self.add_module("af", nn.Linear(self.hidden_dims, 4 * self.n_latents))

    def forward(self, x):
        # batch norm cannot deal with batch_size 1 in train mode
        if self.training and x.size(0) == 1:
            self.eval()
            encoder_outputs = Encoder.forward(self, x)
            self.train()
        else:
            encoder_outputs = Encoder.forward(self, x)
        return encoder_outputs


class DumoulinEncoder(Encoder):
    """ 
    Some Alexnet-inspired encoder with BatchNorm and LeakyReLU as proposed in Dumoulin et al. "Adversarially learned inference"

    Model Architecture (transposed for decoder)
    ------------
    - Convolutional blocks composed of:
        - 1 convolutional layer (2*2^(conv_layer_id) channels), (3 x 3 kernel), (stride of 1), (padding of 1)
        - 1 BatchNorm layer
        - 1 LeakyReLU layer
        - 1 convolutional layer (2*2^(conv_layer_id+1) channels), (4 x 4 kernel), (stride of 2), (padding of 1)
        - 1 BatchNorm layer
        - 1 LeakyReLU layer
    - 1 Convolutional blocks composed of:
        - 1 convolutional layer (2*2^(n_conv_layers) channels), (1 x 1 kernel), (stride of 1), (padding of 1)
        - 1 BatchNorm layer
        - 1 LeakyReLU layer
        - 1 convolutional layer (n_latents channels), (1 x 1 kernel), (stride of 1), (padding of 1)
    """

    def __init__(self, input_shape=(1, 64, 64), n_latents=10, n_conv_layers=4, feature_layer=1, hidden_channels=32, hidden_dims=256,
                 conditional_type="gaussian", use_attention=False):
        Encoder.__init__(self, input_shape=input_shape, n_latents=n_latents, n_conv_layers=n_conv_layers, feature_layer=feature_layer,
                         hidden_channels=hidden_channels, hidden_dims=hidden_dims, conditional_type=conditional_type,
                         use_attention=use_attention)

        # need square and power of 2 image size input
        power = math.log(self.input_shape[1], 2)
        assert (power % 1 == 0.0) and (power > 3), "Dumoulin Encoder needs a power of 2 as image input size (>=16)"
        # need square image input
        assert torch.all(torch.tensor([self.input_shape[i] == self.input_shape[1] for i in range(2, len(self.input_shape))])), "Dumoulin Encoder needs a square image input size"

        assert self.n_conv_layers == power - 2, "The number of convolutional layers in DumoulinEncoder must be log(input_size, 2) - 2 "

        # network architecture
        kernels_size = [4, 4] * self.n_conv_layers
        strides = [1, 2] * self.n_conv_layers
        pads = [0, 1] * self.n_conv_layers
        dils = [1, 1] * self.n_conv_layers

        # feature map size
        feature_map_sizes = conv_output_sizes(self.input_shape[1:], 2 * self.n_conv_layers, kernels_size,
                                              strides, pads,
                                              dils)

        # local feature 
        ## convolutional layers
        self.local_feature_shape = (
            int(self.hidden_channels * math.pow(2, self.feature_layer + 1)),
            feature_map_sizes[2 * self.feature_layer + 1][0],
            feature_map_sizes[2 * self.feature_layer + 1][1])
        self.lf = nn.Sequential()
        for conv_layer_id in range(self.feature_layer + 1):
            if conv_layer_id == 0:
                self.lf.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                    self.conv_module(self.input_shape[0], self.hidden_channels, kernels_size[2 * conv_layer_id],
                                     strides[2 * conv_layer_id], pads[2 * conv_layer_id], dils[2 * conv_layer_id]),
                    self.batchnorm_module(self.hidden_channels),
                    nn.LeakyReLU(inplace=True),
                    self.conv_module(self.hidden_channels, 2 * self.hidden_channels, kernels_size[2 * conv_layer_id + 1],
                                     strides[2 * conv_layer_id + 1], pads[2 * conv_layer_id + 1],
                                     dils[2 * conv_layer_id + 1]),
                    self.batchnorm_module(2 * self.hidden_channels),
                    nn.LeakyReLU(inplace=True)
                ))
            else:
                self.lf.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                    self.conv_module(self.hidden_channels, self.hidden_channels, kernels_size[2 * conv_layer_id],
                                     strides[2 * conv_layer_id], pads[2 * conv_layer_id], dils[2 * conv_layer_id]),
                    self.batchnorm_module(self.hidden_channels),
                    nn.LeakyReLU(inplace=True),
                    self.conv_module(self.hidden_channels, 2 * self.hidden_channels, kernels_size[2 * conv_layer_id + 1],
                                     strides[2 * conv_layer_id + 1], pads[2 * conv_layer_id + 1],
                                     dils[2 * conv_layer_id + 1]),
                    self.batchnorm_module(2 * self.hidden_channels),
                    nn.LeakyReLU(inplace=True)
                ))
            self.hidden_channels *= 2
        self.lf.out_connection_type = ("conv", self.hidden_channels)

        # global feature
        self.gf = nn.Sequential()
        ## convolutional layers
        for conv_layer_id in range(self.feature_layer + 1, self.n_conv_layers):
            self.gf.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                self.conv_module(self.hidden_channels, self.hidden_channels, kernels_size[2 * conv_layer_id],
                                 strides[2 * conv_layer_id],
                                 pads[2 * conv_layer_id], dils[2 * conv_layer_id]),
                self.batchnorm_module(self.hidden_channels),
                nn.LeakyReLU(inplace=True),
                self.conv_module(self.hidden_channels, 2 * self.hidden_channels, kernels_size[2 * conv_layer_id + 1],
                                 strides[2 * conv_layer_id + 1], pads[2 * conv_layer_id + 1],
                                 dils[2 * conv_layer_id + 1]),
                self.batchnorm_module(2 * self.hidden_channels),
                nn.LeakyReLU(inplace=True)
            ))
            self.hidden_channels *= 2
        self.gf.out_connection_type = ("conv", self.hidden_channels)

        # encoding feature
        if self.conditional_type == "gaussian":
            self.add_module("ef", nn.Sequential(
                self.conv_module(self.hidden_channels, self.hidden_channels, kernel_size=1, stride=1),
                self.batchnorm_module(self.hidden_channels),
                nn.LeakyReLU(inplace=True),
                self.conv_module(self.hidden_channels, 2 * self.n_latents, kernel_size=1, stride=1)
            ))
        elif self.conditional_type == "deterministic":
            self.add_module("ef", nn.Sequential(
                self.conv_module(self.hidden_channels, self.hidden_channels, kernel_size=1, stride=1),
                self.batchnorm_module(self.hidden_channels),
                nn.LeakyReLU(inplace=True),
                self.conv_module(self.hidden_channels, self.n_latents, kernel_size=1, stride=1)
            ))

        # attention feature
        if self.use_attention:
            self.add_module("af", nn.Linear(self.hidden_channels, 4 * self.n_latents))

    def forward(self, x):
        # batch norm cannot deal with batch_size 1 in train mode
        if self.training and x.size()[0] == 1:
            self.eval()
            encoder_outputs = Encoder.forward(self, x)
            self.train()
        else:
            encoder_outputs = Encoder.forward(self, x)
        return encoder_outputs

"""==============================================================================================================
DECODERS
==============================================================================================================="""
class Decoder(nn.Module):
    """
    Base Decoder class
    User can specify variable number of convolutional layers to deal with images of different sizes (eg: 3 layers for 32*32 images3 layers for 32*32 images, 6 layers for 256*256 images)

    Parameters
    ----------
    n_channels: number of channels
    input_size: size of input image
    n_conv_layers: desired number of convolutional layers
    n_latents: dimensionality of the infered latent output.
    """

    def __init__(self, output_shape=(1, 64, 64),
                 n_latents=10,
                 n_conv_layers=4,
                 feature_layer=1,
                 hidden_channels=32,
                 hidden_dims=256):
        nn.Module.__init__(self)
        self.output_shape = output_shape
        self.n_latents = n_latents
        self.n_conv_layers = n_conv_layers
        self.feature_layer = feature_layer
        self.hidden_dims = hidden_dims
        self.hidden_channels = hidden_channels

        self.spatial_dims = len(self.output_shape) - 1
        assert 2 <= self.spatial_dims <= 3, "Image must be 2D or 3D"
        if self.spatial_dims == 2:
            self.convtranspose_module = nn.ConvTranspose2d
            self.batchnorm_module = nn.BatchNorm2d
        elif self.spatial_dims == 3:
            self.convtranspose_module = nn.ConvTranspose3d
            self.batchnorm_module = nn.BatchNorm3d

        self.output_keys_list = ["gfi", "lfi", "recon_x"]  # removed z already in encoder outputs

    def forward(self, z):
        if z.dim() == 2 and type(self).__name__ == "DumoulinDecoder":  # B*n_latents -> B*n_latents*1*1
            for _ in range(self.spatial_dims):
                z = z.unsqueeze(-1)

        if bool(torch._C._get_tracing_state()):
            return self.forward_for_graph_tracing(z)

            # global feature map
        gfi = self.efi(z)
        # global feature map
        lfi = self.gfi(gfi)
        # recon_x
        recon_x = self.lfi(lfi)
        # decoder output
        decoder_outputs = {"gfi": gfi, "lfi": lfi, "recon_x": recon_x}

        return decoder_outputs

    def forward_for_graph_tracing(self, z):
        # global feature map
        gfi = self.efi(z)
        # global feature map
        lfi = self.gfi(gfi)
        # recon_x
        recon_x = self.lfi(lfi)
        return recon_x


def get_decoder(model_architecture):
    '''
    model_architecture: string such that the class decoder called is <model_architecture>Decoder
    '''
    return eval("{}Decoder".format(model_architecture))


class BurgessDecoder(Decoder):

    def __init__(self, output_shape=(1, 64, 64), n_latents=10, n_conv_layers=4, feature_layer=1, hidden_channels=32, hidden_dims=256):
        Decoder.__init__(self, output_shape=output_shape, n_latents=n_latents, n_conv_layers=n_conv_layers, feature_layer=feature_layer,
                         hidden_channels=hidden_channels, hidden_dims=hidden_dims)

        # network architecture
        kernels_size = [4] * self.n_conv_layers
        strides = [2] * self.n_conv_layers
        pads = [1] * self.n_conv_layers
        dils = [1] * self.n_conv_layers
        feature_map_sizes = conv_output_sizes(self.output_shape[1:], self.n_conv_layers, kernels_size,
                                              strides, pads, dils)
        n_linear_in = self.hidden_channels * torch.prod(torch.tensor(feature_map_sizes[-1])).item()
        output_pads = [None] * self.n_conv_layers
        output_pads[0] = convtranspose_get_output_padding(feature_map_sizes[0], self.output_shape[1:],
                                                          kernels_size[0],
                                                          strides[0], pads[0])
        for conv_layer_id in range(1, self.n_conv_layers):
            output_pads[conv_layer_id] = convtranspose_get_output_padding(feature_map_sizes[conv_layer_id],
                                                                          feature_map_sizes[conv_layer_id - 1],
                                                                          kernels_size[conv_layer_id],
                                                                          strides[conv_layer_id], pads[conv_layer_id])

        # encoder feature inverse
        self.efi = nn.Sequential(nn.Linear(self.n_latents, self.hidden_dims), nn.ReLU())
        self.efi.out_connection_type = ("lin", self.hidden_dims)

        # global feature inverse
        ## linear layers
        self.gfi = nn.Sequential()
        self.gfi.add_module("lin_1_i", nn.Sequential(nn.Linear(self.hidden_dims, self.hidden_dims), nn.ReLU()))
        self.gfi.add_module("lin_0_i",
                            nn.Sequential(nn.Linear(self.hidden_dims, n_linear_in),
                                          nn.ReLU()))
        self.gfi.add_module("channelize", Channelize(self.hidden_channels, feature_map_sizes[-1]))
        ## convolutional layers
        for conv_layer_id in range(self.n_conv_layers - 1, self.feature_layer + 1 - 1, -1):
            self.gfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(
                self.convtranspose_module(self.hidden_channels, self.hidden_channels, kernels_size[conv_layer_id],
                                          strides[conv_layer_id], pads[conv_layer_id],
                                          output_padding=output_pads[conv_layer_id]), nn.ReLU()))
        self.gfi.out_connection_type = ("conv", self.hidden_channels)

        # local feature inverse
        self.lfi = nn.Sequential()
        for conv_layer_id in range(self.feature_layer + 1 - 1, 0, -1):
            self.lfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(
                self.convtranspose_module(self.hidden_channels, self.hidden_channels, kernels_size[conv_layer_id],
                                          strides[conv_layer_id], pads[conv_layer_id],
                                          output_padding=output_pads[conv_layer_id]), nn.ReLU()))
        self.lfi.add_module("conv_0_i",
                            self.convtranspose_module(self.hidden_channels, self.output_shape[0], kernels_size[0],
                                                      strides[0],
                                                      pads[0],
                                                      output_padding=output_pads[0]))
        self.lfi.out_connection_type = ("conv", self.output_shape[0])


class HjelmDecoder(Decoder):

    def __init__(self, output_shape=(1, 64, 64), n_latents=10, n_conv_layers=4, feature_layer=1, hidden_channels=32, hidden_dims=256):
        Decoder.__init__(self, output_shape=output_shape, n_latents=n_latents, n_conv_layers=n_conv_layers, feature_layer=feature_layer,
                         hidden_channels=hidden_channels, hidden_dims=hidden_dims)

        # network architecture
        kernels_size = [4] * self.n_conv_layers
        strides = [2] * self.n_conv_layers
        pads = [1] * self.n_conv_layers
        dils = [1] * self.n_conv_layers
        feature_map_sizes = conv_output_sizes(self.output_shape[1:], self.n_conv_layers, kernels_size,
                                              strides, pads, dils)
        n_linear_in = self.hidden_channels * torch.prod(torch.tensor(feature_map_sizes[-1])).item()
        output_pads = [None] * self.n_conv_layers
        output_pads[0] = convtranspose_get_output_padding(feature_map_sizes[0], self.output_shape[1:],
                                                          kernels_size[0],
                                                          strides[0], pads[0])
        for conv_layer_id in range(1, self.n_conv_layers):
            output_pads[conv_layer_id] = convtranspose_get_output_padding(feature_map_sizes[conv_layer_id],
                                                                          feature_map_sizes[conv_layer_id - 1],
                                                                          kernels_size[conv_layer_id],
                                                                          strides[conv_layer_id], pads[conv_layer_id])

        # encoder feature inverse
        self.efi = nn.Sequential(nn.Linear(self.n_latents, self.hidden_dims), nn.ReLU())
        self.efi.out_connection_type = ("lin", self.hidden_dims)

        # global feature inverse
        self.hidden_channels = int(self.hidden_channels * math.pow(2, self.n_conv_layers - 1))
        ## linear layers
        self.gfi = nn.Sequential()
        self.gfi.add_module("lin_0_i",
                            nn.Sequential(nn.Linear(self.hidden_dims, n_linear_in),
                                          nn.ReLU()))
        self.gfi.add_module("channelize", Channelize(self.hidden_channels, feature_map_sizes[-1]))
        ## convolutional layers
        for conv_layer_id in range(self.n_conv_layers - 1, self.feature_layer + 1 - 1, -1):
            self.gfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(
                self.convtranspose_module(self.hidden_channels, self.hidden_channels // 2, kernels_size[conv_layer_id],
                                          strides[conv_layer_id], pads[conv_layer_id],
                                          output_padding=output_pads[conv_layer_id]),
                self.batchnorm_module(self.hidden_channels // 2),
                nn.ReLU()))
            self.hidden_channels = self.hidden_channels // 2
        self.gfi.out_connection_type = ("conv", self.hidden_channels)

        # local feature inverse
        self.lfi = nn.Sequential()
        for conv_layer_id in range(self.feature_layer + 1 - 1, 0, -1):
            self.lfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(
                self.convtranspose_module(self.hidden_channels, self.hidden_channels // 2, kernels_size[conv_layer_id],
                                          strides[conv_layer_id], pads[conv_layer_id],
                                          output_padding=output_pads[conv_layer_id]),
                self.batchnorm_module(self.hidden_channels // 2),
                nn.ReLU()))
            self.hidden_channels = self.hidden_channels // 2
        self.lfi.add_module("conv_0_i",
                            self.convtranspose_module(self.hidden_channels, self.output_shape[0], kernels_size[0],
                                                      strides[0],
                                                      pads[0],
                                                      output_padding=output_pads[0]))
        self.lfi.out_connection_type = ("conv", self.output_shape[0])


class DumoulinDecoder(Decoder):

    def __init__(self, output_shape=(1, 64, 64), n_latents=10, n_conv_layers=4, feature_layer=1, hidden_channels=32, hidden_dims=256):
        Decoder.__init__(self, output_shape=output_shape, n_latents=n_latents, n_conv_layers=n_conv_layers, feature_layer=feature_layer,
                         hidden_channels=hidden_channels, hidden_dims=hidden_dims)

        # need square and power of 2 image size input
        power = math.log(self.output_shape[0], 2)
        assert (power % 1 == 0.0) and (power > 3), "Dumoulin Encoder needs a power of 2 as image input size (>=16)"
        assert self.output_shape[0] == self.output_shape[
            1], "Dumoulin Encoder needs a square image input size"

        assert self.n_conv_layers == power - 2, "The number of convolutional layers in DumoulinEncoder must be log(input_size, 2) - 2 "

        # network architecture
        # encoder feature inverse
        self.hidden_channels = int(self.hidden_channels * math.pow(2, self.n_conv_layers))
        kernels_size = [4, 4] * self.n_conv_layers
        strides = [1, 2] * self.n_conv_layers
        pads = [0, 1] * self.n_conv_layers
        dils = [1, 1] * self.n_conv_layers

        feature_map_sizes = conv_output_sizes(self.output_shape[1:], 2 * self.n_conv_layers, kernels_size,
                                              strides, pads,
                                              dils)
        output_pads = [None] * 2 * self.n_conv_layers
        output_pads[0] = convtranspose_get_output_padding(feature_map_sizes[0], self.output_shape[1:],
                                                          kernels_size[0],
                                                          strides[0], pads[0])
        output_pads[1] = convtranspose_get_output_padding(feature_map_sizes[1], feature_map_sizes[0], kernels_size[1],
                                                          strides[1], pads[1])
        for conv_layer_id in range(1, self.n_conv_layers):
            output_pads[2 * conv_layer_id] = convtranspose_get_output_padding(feature_map_sizes[2 * conv_layer_id],
                                                                              feature_map_sizes[
                                                                                  2 * conv_layer_id - 1],
                                                                              kernels_size[2 * conv_layer_id],
                                                                              strides[2 * conv_layer_id],
                                                                              pads[2 * conv_layer_id])
            output_pads[2 * conv_layer_id + 1] = convtranspose_get_output_padding(
                feature_map_sizes[2 * conv_layer_id + 1], feature_map_sizes[2 * conv_layer_id + 1 - 1],
                kernels_size[2 * conv_layer_id + 1], strides[2 * conv_layer_id + 1], pads[2 * conv_layer_id + 1])

        # encoder feature inverse
        self.efi = nn.Sequential(
            self.convtranspose_module(self.n_latents, self.hidden_channels, kernel_size=1, stride=1),
            self.batchnorm_module(self.hidden_channels),
            nn.LeakyReLU(inplace=True),
            self.convtranspose_module(self.hidden_channels, self.hidden_channels, kernel_size=1, stride=1),
            self.batchnorm_module(self.hidden_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.efi.out_connection_type = ("conv", self.hidden_channels)

        # global feature inverse
        self.gfi = nn.Sequential()
        ## convolutional layers
        for conv_layer_id in range(self.n_conv_layers - 1, self.feature_layer + 1 - 1, -1):
            self.gfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(
                self.convtranspose_module(self.hidden_channels, self.hidden_channels // 2, kernels_size[2 * conv_layer_id + 1],
                                          strides[2 * conv_layer_id + 1], pads[2 * conv_layer_id + 1],
                                          output_padding=output_pads[2 * conv_layer_id + 1]),
                self.batchnorm_module(self.hidden_channels // 2),
                nn.LeakyReLU(inplace=True),
                self.convtranspose_module(self.hidden_channels // 2, self.hidden_channels // 2, kernels_size[2 * conv_layer_id],
                                          strides[2 * conv_layer_id], pads[2 * conv_layer_id],
                                          output_padding=output_pads[2 * conv_layer_id]),
                self.batchnorm_module(self.hidden_channels // 2),
                nn.LeakyReLU(inplace=True),
            ))
            self.hidden_channels = self.hidden_channels // 2
        self.gfi.out_connection_type = ("conv", self.hidden_channels)

        # local feature inverse
        self.lfi = nn.Sequential()
        for conv_layer_id in range(self.feature_layer + 1 - 1, 0, -1):
            self.lfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(
                self.convtranspose_module(self.hidden_channels, self.hidden_channels // 2, kernels_size[2 * conv_layer_id + 1],
                                          strides[2 * conv_layer_id + 1], pads[2 * conv_layer_id + 1],
                                          output_padding=output_pads[2 * conv_layer_id + 1]),
                self.batchnorm_module(self.hidden_channels // 2),
                nn.LeakyReLU(inplace=True),
                self.convtranspose_module(self.hidden_channels // 2, self.hidden_channels // 2, kernels_size[2 * conv_layer_id],
                                          strides[2 * conv_layer_id], pads[2 * conv_layer_id],
                                          output_padding=output_pads[2 * conv_layer_id]),
                self.batchnorm_module(self.hidden_channels // 2),
                nn.LeakyReLU(inplace=True),
            ))
            self.hidden_channels = self.hidden_channels // 2
        self.lfi.add_module("conv_0_i", nn.Sequential(
            self.convtranspose_module(self.hidden_channels, self.hidden_channels // 2, kernels_size[1], strides[1], pads[1],
                                      output_padding=output_pads[1]),
            self.batchnorm_module(self.hidden_channels // 2),
            nn.LeakyReLU(inplace=True),
            self.convtranspose_module(self.hidden_channels // 2, self.output_shape[0], kernels_size[0], strides[0],
                                      pads[0],
                                      output_padding=output_pads[0]),
        ))
        self.lfi.out_connection_type = ("conv", self.output_shape[0])

    def forward(self, z):
        # batch norm cannot deal with batch_size 1 in train mode
        if self.training and z.size()[0] == 1:
            self.eval()
            encoder_outputs = Decoder.forward(self, z)
            self.train()
        else:
            encoder_outputs = Decoder.forward(self, z)
        return encoder_outputs

"""==============================================================================================================
LOSSES
==============================================================================================================="""
class BaseLoss():
    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        raise NotImplementedError

def get_loss(loss_name):
    """
    loss_name: string such that the loss called is <loss_name>Loss
    """
    return eval("{}Loss".format(loss_name))

class VAELoss(BaseLoss):
    def __init__(self, reconstruction_dist="bernoulli", **kwargs):
        self.reconstruction_dist = reconstruction_dist

        self.input_keys_list = ['x', 'recon_x', 'mu', 'logvar']

    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        try:
            recon_x = loss_inputs['recon_x']
            mu = loss_inputs['mu']
            logvar = loss_inputs['logvar']
            x = loss_inputs['x']
        except:
            raise ValueError("VAELoss needs {} inputs".format(self.input_keys_list))
        recon_loss = _reconstruction_loss(recon_x, x, self.reconstruction_dist, reduction=reduction)
        KLD_loss, KLD_per_latent_dim, KLD_var = _kld_loss(mu, logvar, reduction=reduction)
        total_loss = recon_loss + KLD_loss

        return {'total': total_loss, 'recon': recon_loss, 'KLD': KLD_loss}


class BetaVAELoss(BaseLoss):
    def __init__(self, beta=5.0, reconstruction_dist="bernoulli", **kwargs):
        self.reconstruction_dist = reconstruction_dist
        self.beta = beta

        self.input_keys_list = ['x', 'recon_x', 'mu', 'logvar']

    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        try:
            recon_x = loss_inputs['recon_x']
            mu = loss_inputs['mu']
            logvar = loss_inputs['logvar']
            x = loss_inputs['x']
        except:
            raise ValueError("BetaVAELoss needs {} inputs".format(self.input_keys_list))
        recon_loss = _reconstruction_loss(recon_x, x, self.reconstruction_dist, reduction=reduction)
        KLD_loss, KLD_per_latent_dim, KLD_var = _kld_loss(mu, logvar, reduction=reduction)
        total_loss = recon_loss + self.beta * KLD_loss

        return {'total': total_loss, 'recon': recon_loss, 'KLD': KLD_loss}


class AnnealedVAELoss(BaseLoss):
    def __init__(self, gamma=1000.0, c_min=0.0, c_max=5.0, c_change_duration=100000, reconstruction_dist="bernoulli",
                 **kwargs):
        self.reconstruction_dist = reconstruction_dist
        self.gamma = gamma
        self.c_min = c_min
        self.c_max = c_max
        self.c_change_duration = c_change_duration

        # update counters
        self.capacity = self.c_min
        self.n_iters = 0

        self.input_keys_list = ['x', 'recon_x', 'mu', 'logvar']

    def update_encoding_capacity(self):
        if self.n_iters > self.c_change_duration:
            self.capacity = self.c_max
        else:
            self.capacity = min(self.c_min + (self.c_max - self.c_min) * self.n_iters / self.c_change_duration,
                                self.c_max)

    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        try:
            recon_x = loss_inputs['recon_x']
            mu = loss_inputs['mu']
            logvar = loss_inputs['logvar']
            x = loss_inputs['x']
        except:
            raise ValueError("AnnealedVAELoss needs {} inputs".format(self.input_keys_list))
        recon_loss = _reconstruction_loss(recon_x, x, self.reconstruction_dist)
        KLD_loss, KLD_per_latent_dim, KLD_var = _kld_loss(mu, logvar)
        total_loss = recon_loss + self.gamma * (KLD_loss - self.capacity).abs()

        if total_loss.requires_grad:  # if we are in "train mode", update counters
            self.n_iters += 1
            self.update_encoding_capacity()

        return {'total': total_loss, 'recon': recon_loss, 'KLD': KLD_loss}
    
def _reconstruction_loss(recon_x, x, reconstruction_dist="bernoulli", reduction="mean"):
    if reconstruction_dist == "bernoulli":
        loss = _bce_with_logits_loss(recon_x, x, reduction=reduction)
    elif reconstruction_dist == "gaussian":
        loss = _mse_loss(recon_x, x, reduction=reduction)
    else:
        raise ValueError("Unkown decoder distribution: {}".format(reconstruction_dist))
    return loss


def _kld_loss(mu, logvar, reduction="mean"):
    """ Returns the KLD loss D(q,p) where q is N(mu,var) and p is N(0,I) """
    if reduction == "mean":
        # 0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_loss_per_latent_dim = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0) / mu.size(
            0)  # we  average on the batch
        # KL-divergence between a diagonal multivariate normal and the standard normal distribution is the sum on each latent dimension
        KLD_loss = torch.sum(KLD_loss_per_latent_dim)
        # we add a regularisation term so that the KLD loss doesnt "trick" the loss by sacrificing one dimension
        KLD_loss_var = torch.var(KLD_loss_per_latent_dim)
    elif reduction == "sum":
        # 0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_loss_per_latent_dim = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
        # KL-divergence between a diagonal multivariate normal and the standard normal distribution is the sum on each latent dimension
        KLD_loss = torch.sum(KLD_loss_per_latent_dim)
        # we add a regularisation term so that the KLD loss doesnt "trick" the loss by sacrificing one dimension
        KLD_loss_var = torch.var(KLD_loss_per_latent_dim)
    elif reduction == "none":
        KLD_loss_per_latent_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        KLD_loss = torch.sum(KLD_loss_per_latent_dim, dim=1)
        KLD_loss_var = torch.var(KLD_loss_per_latent_dim, dim=1)
    else:
        raise ValueError('reduction must be either "mean" | "sum" | "none" ')

    return KLD_loss, KLD_loss_per_latent_dim, KLD_loss_var


def _mse_loss(recon_x, x, reduction="mean"):
    """ Returns the reconstruction loss (mean squared error) summed on the image dims and averaged on the batch size """
    if reduction == "mean":
        mse_loss =  F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
    elif reduction == "sum":
        mse_loss = F.mse_loss(recon_x, x, reduction="sum")
    elif reduction == "none":
        mse_loss = F.mse_loss(recon_x, x, reduction="none")
        mse_loss = mse_loss.view(mse_loss.size(0), -1).sum(1)
    else:
        raise ValueError('reduction must be either "mean" | "sum" | "none" ')
    return mse_loss

def _bce_with_logits_loss(recon_x, x, reduction="mean"):
    """ Returns the reconstruction loss (sigmoid + binary cross entropy) summed on the image dims and averaged on the batch size """
    if reduction == "mean":
        bce_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction="sum") / x.size(0)
    elif reduction == "sum":
        bce_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction="sum")
    elif reduction == "none":
        bce_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction="none")
        bce_loss = bce_loss.view(bce_loss.size(0), -1).sum(1)
    else:
        raise ValueError('reduction must be either "mean" | "sum" | "none" ')
    return bce_loss


"""==============================================================================================================
NN MODULE UTILS 
==============================================================================================================="""


class Flatten(nn.Module):
    """Flatten the input """

    def forward(self, input):
        return input.view(input.size(0), -1)


class Channelize(nn.Module):
    """Channelize a flatten input to the given (C,D,H,W) or (C,H,W) output """

    def __init__(self, n_channels, out_size):
        nn.Module.__init__(self)
        self.n_channels = n_channels
        self.out_size = out_size

    def forward(self, input):
        out_size = (input.size(0), self.n_channels,) + self.out_size
        return input.view(out_size)

def conv_output_sizes(input_size, n_conv=0, kernels_size=1, strides=1, pads=0, dils=1):
    """Returns the size of a tensor after a sequence of convolutions"""
    assert n_conv == len(kernels_size) == len(strides) == len(pads) == len(dils), print(
        'The number of kernels ({}), strides({}), paddings({}) and dilatations({}) has to match the number of convolutions({})'.format(
            len(kernels_size), len(strides), len(pads), len(dils), n_conv))

    spatial_dims = len(input_size)  # 2D or 3D
    in_sizes = list(input_size)
    output_sizes = []

    for conv_id in range(n_conv):
        if type(kernels_size[conv_id]) is not tuple:
            kernel_size = tuple([kernels_size[conv_id]] * spatial_dims)
        if type(strides[conv_id]) is not tuple:
            stride = tuple([strides[conv_id]] * spatial_dims)
        if type(pads[conv_id]) is not tuple:
            pad = tuple([pads[conv_id]] * spatial_dims)
        if type(dils[conv_id]) is not tuple:
            dil = tuple([dils[conv_id]] * spatial_dims)

        for dim in range(spatial_dims):
            in_sizes[dim] = math.floor(
                ((in_sizes[dim] + (2 * pad[dim]) - (dil[dim] * (kernel_size[dim] - 1)) - 1) / stride[dim]) + 1)

        output_sizes.append(tuple(in_sizes))

    return output_sizes


def convtranspose_get_output_padding(input_size, output_size, kernel_size=1, stride=1, pad=0):
    assert len(input_size) == len(output_size)
    spatial_dims = len(input_size)  # 2D or 3D
    out_padding = []

    if type(kernel_size) is not tuple:
        kernel_size = tuple([kernel_size] * spatial_dims)
    if type(stride) is not tuple:
        stride = tuple([stride] * spatial_dims)
    if type(pad) is not tuple:
        pad = tuple([pad] * spatial_dims)

    out_padding = []
    for dim in range(spatial_dims):
        out_padding.append(output_size[dim] + 2 * pad[dim] - kernel_size[dim] - (input_size[dim] - 1) * stride[dim])

    return tuple(out_padding)



"""==============================================================================================================
TENSORBOARD UTILS 
==============================================================================================================="""
def resize_embeddings(embedding_images, sprite_size=8192):
    image_size = max(embedding_images.shape[-2], embedding_images.shape[-1]) #show on last 2 dims (HW)
    n_images = np.ceil(np.sqrt(len(embedding_images)))
    if n_images * image_size <= sprite_size:
        return embedding_images
    else:
        image_ratio = sprite_size / (n_images * image_size)
        return F.interpolate(embedding_images, size=int(image_size*image_ratio))

def logger_add_image_list(logger, image_list, tag, global_step=0, nrow=None, padding=0, n_channels=1, spatial_dims=2):
    if isinstance(image_list, list):
        image_tensor = torch.stack(image_list)
    elif isinstance(image_list, torch.Tensor):
        image_tensor = image_list
    if nrow is None:
        nrow = int(np.sqrt(image_tensor.shape[0]))
    if padding is None:
        padding = 0

    if nrow == 0:
        return

    if n_channels == 1 or n_channels == 3:  # grey scale or RGB
        if spatial_dims == 2:
            img = make_grid(image_tensor, nrow=nrow, padding=padding)
            logger.add_image(tag, img, global_step=global_step)
        elif spatial_dims == 3:
            logger.add_video(tag, image_tensor, global_step=global_step, dataformats="NCTHW")
        else:
            raise NotImplementedError
    else:
        if spatial_dims == 2:
            img = make_grid(image_tensor.argmax(1).unsqueeze(1).float()/n_channels, nrow=nrow, padding=padding)
            logger.add_image(tag, img, global_step=global_step)
        elif spatial_dims == 3:
            logger.add_video(tag, image_tensor.argmax(1).unsqueeze(1).float()/n_channels, global_step=global_step, dataformats="NCTHW")
        else:
            raise NotImplementedError
    return


class ExperimentHistoryDataset(Dataset):
    """ Represents an abstract dataset that uses the Experiment DB History.

    Input params:
        transform: PyTorch transform to apply on-the-fly to every data tensor instance (default=None).
    """


    def __init__(self, access_history_fn, history_ids, wrapped_input_space_key, transform=None, **kwargs):


        self.access_history_fn = access_history_fn
        self.history_ids = history_ids
        self.wrapped_input_space_key = wrapped_input_space_key
        self.transform = transform

    def __len__(self):
        return len(self.history_ids)

    def __getitem__(self, idx):
        rel_idx = self.history_ids[idx]
        data = self.access_history_fn()['input'][rel_idx][self.wrapped_input_space_key]

        if self.transform is not None:
            data = self.transform(data)

        return {"obs": data, "label": torch.Tensor([-1]) , "index": idx}