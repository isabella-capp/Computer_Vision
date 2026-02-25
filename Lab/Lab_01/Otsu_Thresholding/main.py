import random
import numpy as np
from skimage import data
from skimage.transform import resize
import torch

im = data.camera()
im = resize(im, (im.shape[0] // 2, im.shape[1] // 2), mode='reflect', preserve_range=True, anti_aliasing=True)
im = np.asarray(im, dtype=np.uint8)
im = torch.from_numpy(im)

hist = torch.histc(im.float(), bins=256, min=0, max=255)
probs = hist / torch.numel(im)

max_variance = 0
optimal_threshold = 0

for t in range(256):
    q1 = torch.sum(probs[:t+1])
    q2 = torch.sum(probs[t+1:])
    
    if q1 == 0 or q2 == 0:
        continue
    
    mu_1 = torch.sum(torch.arange(t+1, dtype=torch.float32) * probs[:t+1]) / q1
    mu_2 = torch.sum(torch.arange(t+1, 256, dtype=torch.float32) * probs[t+1:]) / q2
    
    # Between-class variance
    variance = q1 * q2 * (mu_1 - mu_2) ** 2
    
    if variance > max_variance:
        max_variance = variance
        optimal_threshold = t

out = optimal_threshold

