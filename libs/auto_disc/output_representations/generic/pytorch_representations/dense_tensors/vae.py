from auto_disc.output_representations import BaseOutputRepresentation
from auto_disc.output_representations.generic.pytorch_representations.dense_tensors.trunks.encoders import get_encoder
from auto_disc.output_representations.generic.pytorch_representations.dense_tensors.heads.decoders import get_decoder
from auto_disc.utils.config_parameters import IntegerConfigParameter, StringConfigParameter, BooleanConfigParameter
from auto_disc.utils.misc.torch_utils import ExperimentHistoryDataset
from auto_disc.utils.misc.tensorboard_utils import logger_add_image_list, resize_embeddings
from auto_disc.utils.spaces.utils import ConfigParameterBinding
from auto_disc.utils.spaces import DictSpace, BoxSpace
import os
import sys
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


#@StringConfigParameter(name="input_tensors_layout", default="dense", possible_values=["dense", "minkowski", ])
@StringConfigParameter(name="input_tensors_device", default="cpu", possible_values=["cuda", "cpu", ])
# TODO: Add device as part of spaces (?)

@StringConfigParameter(name="encoder_name", default="Burgess", possible_values=["Burgess", "Dumoulin", ])
@IntegerConfigParameter(name="encoder_n_latents", default=10, min=1)
@IntegerConfigParameter(name="encoder_n_conv_layers", default=4, min=1)
@IntegerConfigParameter(name="encoder_feature_layer", default=1, min=1)
@IntegerConfigParameter(name="encoder_hidden_channels", default=32, min=1)
@IntegerConfigParameter(name="encoder_hidden_dims", default=256, min=1)
@StringConfigParameter(name="encoder_conditional_type", default="gaussian",
                       possible_values=["gaussian", "deterministic", ])
@BooleanConfigParameter(name="encoder_use_attention", default=False)
@StringConfigParameter(name="weights_init_name", default="pytorch", possible_values=["pytorch", ])
# TODO: DictConfigParameter for weights_init_parameters

@StringConfigParameter(name="loss_name", default="VAE", possible_values=["VAE", "betaVAE", "annealedVAE", ])
# TODO: DictConfigParameter for loss_parameters

@StringConfigParameter(name="optimizer_name", default="Adam", possible_values=["Adam", ])
# TODO: DictConfigParameter for optimizer_parameters

@BooleanConfigParameter(name="checkpoint_active", default=True)
# TODO: replace checkpoint frequency with callbacks
@StringConfigParameter(name="checkpoint_folder", default="./checkpoints", possible_values="all")
@BooleanConfigParameter(name="checkpoint_best_model", default=False)
@IntegerConfigParameter(name="checkpoint_frequency", default=10, min=1)

#@BooleanConfigParameter(name="tensorboard_active", default=True)
# TODO: replace tensorboard frequency with callbacks
@StringConfigParameter(name="tb_folder", default="./tensorboard", possible_values="all")
@IntegerConfigParameter(name="tb_record_loss_frequency", default=1, min=1)
@IntegerConfigParameter(name="tb_record_images_frequency", default=10, min=1)
@IntegerConfigParameter(name="tb_record_embeddings_frequency", default=10, min=1)
@IntegerConfigParameter(name="tb_record_memory_max", default=100, min=1)
@IntegerConfigParameter(name="train_period", default=10, min=1)


class VAE(nn.Module, BaseOutputRepresentation):
    """
    VAE Output Representation
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
        encoder_cls = get_encoder(self.config.encoder_name)
        self.encoder = encoder_cls(input_shape=self.input_space[self.wrapped_input_space_key].shape,
                                   n_latents=self.config.encoder_n_latents,
                                   n_conv_layers=self.config.encoder_n_conv_layers,
                                   feature_layer=self.config.encoder_feature_layer,
                                   hidden_channels=self.config.encoder_hidden_channels,
                                   hidden_dims=self.config.encoder_hidden_dims,
                                   conditional_type=self.config.encoder_conditional_type,
                                   use_attention=self.config.encoder_use_attention,
                                   )
        decoder_cls = get_decoder(self.config.encoder_name)
        self.decoder = decoder_cls(output_shape=self.input_space[self.wrapped_input_space_key].shape,
                                   n_latents=self.config.encoder_n_latents,
                                   n_conv_layers=self.config.encoder_n_conv_layers,
                                   feature_layer=self.config.encoder_feature_layer,
                                   hidden_channels=self.config.encoder_hidden_channels,
                                   hidden_dims=self.config.encoder_hidden_dims,
                                   )
        self.init_network_weights()

    def init_network_weights(self):
        # TODO
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
        # TODO: option for no scheduler
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
            dummy_input = torch.Tensor(size=self.input_space[self.wrapped_input_space_key].shape).uniform_(0, 1) \
                .type(self.input_space[self.wrapped_input_space_key].dtype) \
                .to(self.config.input_tensors_device) \
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
                # TODO: proper save function with AutoDiscTool, torch.save throws error here
                # torch.save(self, os.path.join(self.config.checkpoint_folder, 'current_weight_model.pth'))
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
                x = data['obs'].to(self.config.input_tensors_device) \
                    .type(self.input_space[self.wrapped_input_space_key].dtype)
                # TODO: see why the dtype is modified through TinyDB?
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
                x = data['obs'].to(self.config.input_tensors_device) \
                    .type(self.input_space[self.wrapped_input_space_key].dtype)
                # TODO: see why the dtype is modified through TinyDB?
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
                        embeddings.append(
                            model_outputs["z"].cpu().detach().view(x.shape[0], self.config.encoder_n_latents))
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
                                  global_step=self.n_epochs,
                                  n_channels=self.input_space[self.wrapped_input_space_key].shape[0],
                                  spatial_dims=len(self.input_space[self.wrapped_input_space_key].shape[1:]))

        if record_embeddings:
            if len(images.shape) == 5:
                images = images[:, :, self.input_space[self.wrapped_input_space_key].shape[0] // 2, :,
                         :]  # we take slice at middle depth only
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
                                                 history_ids=list(range(-5, 0)),
                                                 wrapped_input_space_key=self.wrapped_input_space_key)
        valid_loader = DataLoader(valid_dataset)
        self.run_training(train_loader=train_loader, n_epochs=10, valid_loader=valid_loader)

    def map(self, observations, is_output_new_discovery, **kwargs):

        if (self.CURRENT_RUN_INDEX % self.config.train_period == 0) and (
                self.CURRENT_RUN_INDEX > 0) and is_output_new_discovery:
            self.train_update()

        input = observations[self.wrapped_input_space_key] \
            .to(self.config.input_tensors_device) \
            .type(self.input_space[self.wrapped_input_space_key].dtype) \
            .unsqueeze(0)
        output = self.encoder.calc_embedding(input).flatten().cpu().detach()

        return {"embedding": output}


"""==============================================================================================================
LOSSES
==============================================================================================================="""
class VAELoss:
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


class BetaVAELoss:
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


class AnnealedVAELoss:
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
        mse_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
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