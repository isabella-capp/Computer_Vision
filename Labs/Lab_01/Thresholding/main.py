import random
import numpy as np
from skimage import data
from skimage.transform import resize
import torch

im = data.camera()
im = resize(im, (im.shape[0] // 2, im.shape[1] // 2), mode='reflect', preserve_range=True, anti_aliasing=True)
im = np.asarray(im, dtype=np.uint8)
im = torch.from_numpy(im)
val = random.randint(0, 255)

out = torch.where(im > val, 255, 0).to(torch.uint8)