from auto_disc.utils.misc.torch_utils import Channelize, conv_output_sizes, convtranspose_get_output_padding
import math
import torch
from torch import nn


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

        # global feature map
        gfi = self.efi(z)
        # global feature map
        lfi = self.gfi(gfi)
        # recon_x
        recon_x = self.lfi(lfi)
        # decoder output
        decoder_outputs = {"gfi": gfi, "lfi": lfi, "recon_x": recon_x}

        return decoder_outputs


def get_decoder(model_architecture):
    '''
    model_architecture: string such that the class decoder called is <model_architecture>Decoder
    '''
    return eval("{}Decoder".format(model_architecture))


class BurgessDecoder(Decoder):

    def __init__(self, output_shape=(1, 64, 64), n_latents=10, n_conv_layers=4, feature_layer=1, hidden_channels=32,
                 hidden_dims=256):
        Decoder.__init__(self, output_shape=output_shape, n_latents=n_latents, n_conv_layers=n_conv_layers,
                         feature_layer=feature_layer,
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

    def __init__(self, output_shape=(1, 64, 64), n_latents=10, n_conv_layers=4, feature_layer=1, hidden_channels=32,
                 hidden_dims=256):
        Decoder.__init__(self, output_shape=output_shape, n_latents=n_latents, n_conv_layers=n_conv_layers,
                         feature_layer=feature_layer,
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

    def __init__(self, output_shape=(1, 64, 64), n_latents=10, n_conv_layers=4, feature_layer=1, hidden_channels=32,
                 hidden_dims=256):
        Decoder.__init__(self, output_shape=output_shape, n_latents=n_latents, n_conv_layers=n_conv_layers,
                         feature_layer=feature_layer,
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
                self.convtranspose_module(self.hidden_channels, self.hidden_channels // 2,
                                          kernels_size[2 * conv_layer_id + 1],
                                          strides[2 * conv_layer_id + 1], pads[2 * conv_layer_id + 1],
                                          output_padding=output_pads[2 * conv_layer_id + 1]),
                self.batchnorm_module(self.hidden_channels // 2),
                nn.LeakyReLU(inplace=True),
                self.convtranspose_module(self.hidden_channels // 2, self.hidden_channels // 2,
                                          kernels_size[2 * conv_layer_id],
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
                self.convtranspose_module(self.hidden_channels, self.hidden_channels // 2,
                                          kernels_size[2 * conv_layer_id + 1],
                                          strides[2 * conv_layer_id + 1], pads[2 * conv_layer_id + 1],
                                          output_padding=output_pads[2 * conv_layer_id + 1]),
                self.batchnorm_module(self.hidden_channels // 2),
                nn.LeakyReLU(inplace=True),
                self.convtranspose_module(self.hidden_channels // 2, self.hidden_channels // 2,
                                          kernels_size[2 * conv_layer_id],
                                          strides[2 * conv_layer_id], pads[2 * conv_layer_id],
                                          output_padding=output_pads[2 * conv_layer_id]),
                self.batchnorm_module(self.hidden_channels // 2),
                nn.LeakyReLU(inplace=True),
            ))
            self.hidden_channels = self.hidden_channels // 2
        self.lfi.add_module("conv_0_i", nn.Sequential(
            self.convtranspose_module(self.hidden_channels, self.hidden_channels // 2, kernels_size[1], strides[1],
                                      pads[1],
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

class ConnectedDecoder(Decoder):
    def __init__(self, decoder_instance, connect_gfi=False, connect_lfi=False, connect_recon=False):
        Decoder.__init__(self, output_shape=decoder_instance.output_shape,
                             n_latents=decoder_instance.n_latents,
                             n_conv_layers=decoder_instance.n_conv_layers,
                             feature_layer=decoder_instance.feature_layer,
                             hidden_channels=decoder_instance.hidden_channels,
                             hidden_dims=decoder_instance.hidden_dims,
                         )

        self.connect_gfi = connect_gfi
        self.connect_lfi = connect_lfi
        self.connect_recon = connect_recon

        # copy parent network layers
        self.efi = decoder_instance.efi
        self.gfi = decoder_instance.gfi
        self.lfi = decoder_instance.lfi

        self.spatial_dims = decoder_instance.spatial_dims
        if self.spatial_dims == 2:
            self.conv_module = nn.Conv2d
        elif self.spatial_dims == 3:
            self.conv_module = nn.Conv3d

        # add lateral connections

        ## gfi
        if self.connect_gfi:
            if self.efi.out_connection_type[0] == "conv":
                connection_channels = self.efi.out_connection_type[1]
                # self.gfi_c = nn.Sequential(self.conv_module(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False), nn.ReLU())
                self.gfi_c_beta = self.conv_module(connection_channels, connection_channels, kernel_size=1,
                                                   stride=1, bias=False)
                self.gfi_c_gamma = self.conv_module(connection_channels, connection_channels, kernel_size=1,
                                                    stride=1, bias=False)
            elif self.efi.out_connection_type[0] == "lin":
                connection_dim = self.efi.out_connection_type[1]
                # self.gfi_c = nn.Sequential(nn.Linear(connection_dim, connection_dim), nn.ReLU())
                self.gfi_c_beta = nn.Linear(connection_dim, connection_dim)
                self.gfi_c_gamma = nn.Linear(connection_dim, connection_dim)
        ## lfi
        if self.connect_lfi:
            if self.gfi.out_connection_type[0] == "conv":
                connection_channels = self.gfi.out_connection_type[1]
                # self.lfi_c = nn.Sequential(self.conv_module(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False), nn.ReLU())
                self.lfi_c_beta = self.conv_module(connection_channels, connection_channels, kernel_size=1,
                                                   stride=1, bias=False)
                self.lfi_c_gamma = self.conv_module(connection_channels, connection_channels, kernel_size=1,
                                                    stride=1, bias=False)
            elif self.gfi.out_connection_type[0] == "lin":
                connection_dim = self.gfi.out_connection_type[1]
                # self.lfi_c = nn.Sequential(nn.Linear(connection_dim, connection_dim), nn.ReLU())
                self.lfi_c_beta = nn.Linear(connection_dim, connection_dim)
                self.lfi_c_gamma = nn.Linear(connection_dim, connection_dim)

        ## recon
        if self.connect_recon:
            if self.lfi.out_connection_type[0] == "conv":
                connection_channels = self.lfi.out_connection_type[1]
                # self.recon_c = nn.Sequential(self.conv_module(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False))
                self.recon_c_beta = self.conv_module(connection_channels, connection_channels, kernel_size=1,
                                                     stride=1, bias=False)
                self.recon_c_gamma = self.conv_module(connection_channels, connection_channels, kernel_size=1,
                                                      stride=1, bias=False)
            elif self.lfi.out_connection_type[0] == "lin":
                connection_dim = self.lfi.out_connection_type[1]
                # self.recon_c = nn.Sequential(nn.Linear(connection_dim, connection_dim), nn.ReLU())
                self.recon_c_beta = nn.Linear(connection_dim, connection_dim)
                self.recon_c_gamma = nn.Linear(connection_dim, connection_dim)

        #TODO: weight initialisation

    def forward(self, z, parent_gfi=None, parent_lfi=None, parent_recon_x=None):

        if z.dim() == 2 and self.efi.out_connection_type[0] == "conv":  # B*n_latents -> B*n_latents*1*1(*1)
            for _ in range(self.spatial_dims):
                z = z.unsqueeze(-1)

        # batch norm cannot deal with batch_size 1 in train mode
        was_training = None
        if self.training and z.size()[0] == 1:
            self.eval()
            was_training = True

        # global feature map
        gfi = self.efi(z)
        # add the connections
        if self.connect_gfi:
            # gfi = gfi + self.gfi_c(parent_gfi)
            gfi = gfi * self.gfi_c_gamma(parent_gfi) + self.gfi_c_beta(parent_gfi)

        # local feature map
        lfi = self.gfi(gfi)
        # add the connections
        if self.connect_lfi:
            # lfi = lfi + self.lfi_c(parent_lfi)
            lfi = lfi * self.lfi_c_gamma(parent_lfi) + self.lfi_c_beta(parent_lfi)

        # recon_x
        recon_x = self.lfi(lfi)
        # add the connections
        if self.connect_recon:
            # recon_x = recon_x + self.recon_c(parent_recon_x)
            recon_x = recon_x * self.recon_c_gamma(parent_recon_x) + self.recon_c_beta(parent_recon_x)

        # decoder output
        decoder_outputs = {"gfi": gfi, "lfi": lfi, "recon_x": recon_x}

        if was_training and z.size()[0] == 1:
            self.train()

        return decoder_outputs