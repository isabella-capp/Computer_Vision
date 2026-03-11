import random
import numpy as np
import torch
from skimage import data


def color_histogram(im, nbin):
    """Compute normalized color histogram for RGB image.
    
    Args:
        im: Input image tensor of shape (3, H, W)
        nbin: Number of bins for histogram
    
    Returns:
        Normalized concatenated histogram for all channels
    """
    im_quantized = torch.floor(im.long() * nbin / 256)
    hist0 = torch.histc(im_quantized[0].float(), bins=nbin, min=0, max=nbin-1)
    hist1 = torch.histc(im_quantized[1].float(), bins=nbin, min=0, max=nbin-1)
    hist2 = torch.histc(im_quantized[2].float(), bins=nbin, min=0, max=nbin-1)
    
    concatenated_hist = torch.cat([hist0, hist1, hist2])
    normalized_hist = concatenated_hist / torch.sum(concatenated_hist)
    
    return normalized_hist


def main():
    im = data.astronaut()
    im = np.swapaxes(np.swapaxes(im, 0, 2), 1, 2)
    im = torch.from_numpy(im)
    nbin = random.randint(32, 128)
    
    out = color_histogram(im, nbin)
    print(out)


if __name__ == "__main__":
    main()