import numpy as np
from matplotlib import colors
import torch
from torchvision.utils import make_grid
import torch.nn.functional as F

def resize_embeddings(embedding_images, sprite_size=8192):
    batch_size = embedding_images.shape[0]
    n_channels = embedding_images.shape[1]
    img_shape = embedding_images.shape[2:]
    if len(img_shape) == 3:
        embedding_images = embedding_images[:, :, img_shape[0] // 2, :, :]  # we take slice at middle depth only
        img_shape = embedding_images.shape[2:]
    if (n_channels != 1) or (n_channels != 3):
        channel_colors = [colors.to_rgb(color) for color in colors.TABLEAU_COLORS.values()][:n_channels]
        channel_colors = torch.as_tensor(channel_colors).t().type(embedding_images.dtype)  # shape(3,n_channels)
        embedding_images = (channel_colors @ embedding_images.view(batch_size, n_channels, -1)).view(batch_size, 3, *img_shape)

    image_size = max(img_shape)
    n_images = np.ceil(np.sqrt(batch_size))
    if n_images * image_size <= sprite_size:
        return embedding_images
    else:
        image_ratio = sprite_size / (n_images * image_size)
        return F.interpolate(embedding_images, size=int(image_size*image_ratio))

def logger_add_image_list(logger, image_list, tag, global_step=0, nrow=None, padding=0):
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

    batch_size = image_tensor.shape[0]
    n_channels = image_tensor.shape[1]
    img_shape = image_tensor.shape[2:]
    spatial_dims = len(img_shape)

    if n_channels != 1 or n_channels != 3:  # nor grey scale nor RGB, convert to RGB from list of color per channel
        channel_colors = [colors.to_rgb(color) for color in colors.TABLEAU_COLORS.values()][:n_channels]
        channel_colors = torch.as_tensor(channel_colors).t().type(image_tensor.dtype)  # shape(3,n_channels)
        image_tensor = (channel_colors @ image_tensor.view(batch_size, n_channels, -1)).view(batch_size, 3, *img_shape)

    if spatial_dims == 2:
        img = make_grid(image_tensor, nrow=nrow, padding=padding)
        logger.add_image(tag, img, global_step=global_step)
    elif spatial_dims == 3:
        logger.add_video(tag, image_tensor, global_step=global_step, dataformats="NCTHW")
    else:
        raise NotImplementedError
    return