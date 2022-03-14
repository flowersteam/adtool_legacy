import os
from PIL import Image
import torch
import numpy as np

img = Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "gecko.png"))
img = img.resize((256, 256), Image.ANTIALIAS)

with torch.no_grad():
    img = torch.as_tensor(np.float32(img)) / 255.0
    # premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]
    rgb, a = img[..., :3], img[..., 3:4].clamp(0.0, 1.0)
    img = 1.0 - a + rgb
    gray_target_img = img.matmul(torch.FloatTensor([[0.2989, 0.5870, 0.1140]]).t()).squeeze()
    target_img = (1.0 - gray_target_img).flatten().unsqueeze(0)