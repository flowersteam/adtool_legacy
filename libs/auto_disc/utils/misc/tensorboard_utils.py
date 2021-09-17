import numpy as np
import torch
from torchvision.utils import make_grid
import torch.nn.functional as F

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