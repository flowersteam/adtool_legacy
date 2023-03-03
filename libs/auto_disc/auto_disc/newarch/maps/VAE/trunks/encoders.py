from auto_disc.utils.misc.torch_utils import Flatten, conv_output_sizes
from collections import namedtuple
import math
import torch
from torch import nn
import torch.nn.functional as F

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

    def __init__(self, input_shape=(1, 64, 64),
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

        output_keys_list = ["x", "lf", "gf", "z"]
        if self.conditional_type == "gaussian":
            output_keys_list += ["mu", "logvar"]
        if self.use_attention:
            output_keys_list.append("af")
        self.output_class = namedtuple("output", output_keys_list)

    def forward(self, x):

        # local feature map
        lf = self.lf(x)
        # global feature map
        gf = self.gf(lf)


        # encoding
        if self.conditional_type == "gaussian":
            mu, logvar = torch.chunk(self.ef(gf), 2, dim=1)
            z = self.reparameterize(mu, logvar)
            if z.ndim > 2:
                for _ in range(self.spatial_dims):
                    mu = mu.squeeze(dim=-1)
                    logvar = logvar.squeeze(dim=-1)
                    z = z.squeeze(dim=-1)

        elif self.conditional_type == "deterministic":
            z = self.ef(gf)
            if z.ndim > 2:
                for _ in range(self.spatial_dims):
                    z = z.squeeze(dim=-1)

        # attention features
        if self.use_attention:
            af = self.af(gf)
            af = F.normalize(af, p=2)

        keys = self.output_class._fields
        values = []
        for k in keys:
            values.append(eval(k))
        outputs = self.output_class(**dict(zip(keys, values)))

        return outputs

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def calc_embedding(self, x):
        outputs = self.forward(x)
        return outputs.z


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

    def __init__(self, input_shape=(1, 64, 64), n_latents=10, n_conv_layers=4, feature_layer=1, hidden_channels=32,
                 hidden_dims=256, conditional_type="gaussian", use_attention=False):
        Encoder.__init__(self, input_shape=input_shape, n_latents=n_latents, n_conv_layers=n_conv_layers,
                         feature_layer=feature_layer, hidden_channels=hidden_channels, hidden_dims=hidden_dims,
                         conditional_type=conditional_type, use_attention=use_attention)

        # need square image input
        assert torch.all(torch.tensor([self.input_shape[i] == self.input_shape[1] for i in range(2,
                                                                                                 len(self.input_shape))])), "BurgessEncoder needs a square image input size"

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
                self.conv_module(self.hidden_channels, self.hidden_channels, kernels_size[conv_layer_id],
                                 strides[conv_layer_id],
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

    def __init__(self, input_shape=(1, 64, 64), n_latents=10, n_conv_layers=4, feature_layer=1, hidden_channels=32,
                 hidden_dims=256,
                 conditional_type="gaussian", use_attention=False):
        Encoder.__init__(self, input_shape=input_shape, n_latents=n_latents, n_conv_layers=n_conv_layers,
                         feature_layer=feature_layer,
                         hidden_channels=hidden_channels, hidden_dims=hidden_dims, conditional_type=conditional_type,
                         use_attention=use_attention)

        # need square image input
        assert torch.all(torch.tensor([self.input_shape[i] == self.input_shape[1] for i in range(2,
                                                                                                 len(self.input_shape))])), "HjelmEncoder needs a square image input size"

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
                                 pads[conv_layer_id], dils[conv_layer_id]),
                self.batchnorm_module(self.hidden_channels * 2),
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
            outputs = Encoder.forward(self, x)
            self.train()
        else:
            outputs = Encoder.forward(self, x)
        return outputs


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

    def __init__(self, input_shape=(1, 64, 64), n_latents=10, n_conv_layers=4, feature_layer=1, hidden_channels=32,
                 hidden_dims=256,
                 conditional_type="gaussian", use_attention=False):
        Encoder.__init__(self, input_shape=input_shape, n_latents=n_latents, n_conv_layers=n_conv_layers,
                         feature_layer=feature_layer,
                         hidden_channels=hidden_channels, hidden_dims=hidden_dims, conditional_type=conditional_type,
                         use_attention=use_attention)

        # need square and power of 2 image size input
        power = math.log(self.input_shape[1], 2)
        assert (power % 1 == 0.0) and (power > 3), "Dumoulin Encoder needs a power of 2 as image input size (>=16)"
        # need square image input
        assert torch.all(torch.tensor([self.input_shape[i] == self.input_shape[1] for i in range(2,
                                                                                                 len(self.input_shape))])), "Dumoulin Encoder needs a square image input size"

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
                    self.conv_module(self.hidden_channels, 2 * self.hidden_channels,
                                     kernels_size[2 * conv_layer_id + 1],
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
                    self.conv_module(self.hidden_channels, 2 * self.hidden_channels,
                                     kernels_size[2 * conv_layer_id + 1],
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
            outputs = Encoder.forward(self, x)
            self.train()
        else:
            outputs = Encoder.forward(self, x)
        return outputs


class ConnectedEncoder(Encoder):
    def __init__(self, encoder_instance, connect_lf=False, connect_gf=False, **kwargs):
        Encoder.__init__(self, input_shape=encoder_instance.input_shape,
                             n_latents=encoder_instance.n_latents,
                             n_conv_layers=encoder_instance.n_conv_layers,
                             feature_layer=encoder_instance.feature_layer,
                             hidden_channels=encoder_instance.hidden_channels,
                             hidden_dims=encoder_instance.hidden_dims,
                             conditional_type=encoder_instance.conditional_type,
                             use_attention=encoder_instance.use_attention,
                         )
        self.connect_lf = connect_lf
        self.connect_gf = connect_gf

        # copy parent network layers
        self.lf = encoder_instance.lf
        self.gf = encoder_instance.gf
        self.ef = encoder_instance.ef

        # add lateral connections
        self.spatial_dims = encoder_instance.spatial_dims
        if self.spatial_dims == 2:
            self.conv_module = nn.Conv2d
        elif self.spatial_dims == 3:
            self.conv_module = nn.Conv3d
        ## lf
        if self.connect_lf:
            if self.lf.out_connection_type[0] == "conv":
                connection_channels = self.lf.out_connection_type[1]
                #self.lf_c = nn.Sequential(self.conv_module(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False), nn.ReLU())
                self.lf_c_beta = self.conv_module(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False)
                self.lf_c_gamma = self.conv_module(connection_channels, connection_channels, kernel_size=1, stride=1, bias=False)
            elif self.lf.out_connection_type[0] == "lin":
                connection_dim = self.lf.out_connection_type[1]
                #self.lf_c = nn.Sequential(nn.Linear(connection_dim, connection_dim), nn.ReLU())
                self.lf_c_beta = nn.Linear(connection_dim, connection_dim)
                self.lf_c_gamma = nn.Linear(connection_dim, connection_dim)

        ## gf
        if self.connect_gf:
            if self.gf.out_connection_type[0] == "conv":
                connection_channels = self.gf.out_connection_type[1]
                #self.gf_c = nn.Sequential(self.conv_module(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False), nn.ReLU())
                self.gf_c_beta = self.conv_module(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False)
                self.gf_c_gamma = self.conv_module(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False)
            elif self.gf.out_connection_type[0] == "lin":
                connection_dim = self.gf.out_connection_type[1]
                #self.gf_c = nn.Sequential(nn.Linear(connection_dim, connection_dim), nn.ReLU())
                self.gf_c_beta = nn.Linear(connection_dim, connection_dim)
                self.gf_c_gamma = nn.Linear(connection_dim, connection_dim)

        #TODO: weight initialization


    def forward(self, x, parent_lf=None, parent_gf=None):

        # batch norm cannot deal with batch_size 1 in train mode
        was_training = None
        if self.training and x.size()[0] == 1:
            self.eval()
            was_training = True


        # local feature map
        lf = self.lf(x)
        # add the connections
        if self.connect_lf:
            #lf = lf + self.lf_c(parent_lf)
            lf = lf * self.lf_c_gamma(parent_lf) + self.lf_c_beta(parent_lf)

        # global feature map
        gf = self.gf(lf)
        # add the connections
        if self.connect_gf:
            #gf = gf + self.gf_c(parent_gf)
            gf = gf * self.gf_c_gamma(parent_gf) + self.gf_c_beta(parent_gf)

        # encoding
        if self.conditional_type == "gaussian":
            mu, logvar = torch.chunk(self.ef(gf), 2, dim=1)
            z = self.reparameterize(mu, logvar)
            if z.ndim > 2:
                for _ in range(self.spatial_dims):
                    mu = mu.squeeze(-1)
                    logvar = logvar.squeeze(-1)
                    z = z.squeeze(-1)

        elif self.conditional_type == "deterministic":
            z = self.ef(gf)
            if z.ndim > 2:
                for _ in range(self.spatial_dims):
                    z = z.squeeze(-1)

        if was_training and x.size()[0] == 1:
            self.train()

        keys = self.output_class._fields
        values = []
        for k in keys:
            values.append(eval(k))
        outputs = self.output_class(**dict(zip(keys, values)))

        return outputs
