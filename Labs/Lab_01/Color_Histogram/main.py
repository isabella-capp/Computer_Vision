import random
import numpy as np
import torch
from skimage import data

im = data.astronaut()
im = np.swapaxes(np.swapaxes(im, 0, 2), 1, 2)
im = torch.from_numpy(im)
nbin = random.randint(32,128)

im_quantized = torch.floor(im.long() * nbin / 256)
hist0 = torch.histc(im_quantized[0].float(), bins=nbin, min=0, max=nbin-1)
hist1 = torch.histc(im_quantized[1].float(), bins=nbin, min=0, max=nbin-1)
hist2 = torch.histc(im_quantized[2].float(), bins=nbin, min=0, max=nbin-1)


concatenated_hist = torch.cat([hist0, hist1, hist2])
normalized_hist = concatenated_hist / torch.sum(concatenated_hist)

out = normalized_hist