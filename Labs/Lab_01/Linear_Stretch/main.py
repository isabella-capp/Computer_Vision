import random
import numpy as np
import torch
from skimage import data
from skimage.transform import resize


def linear_stretch(im, a, b):
    """Apply linear stretch transformation to image.
    
    Args:
        im: Input image tensor
        a: Multiplicative factor
        b: Additive factor
    
    Returns:
        Stretched image clamped to [0, 255]
    """
    out = torch.round(a * im + b)
    return out.clamp(0, 255).to(torch.uint8)


def main():
    im = data.coffee()
    im = resize(im, (im.shape[0] // 8, im.shape[1] // 8), mode='reflect', preserve_range=True, anti_aliasing=True)
    im = np.asarray(im, dtype=np.uint8)
    im = np.swapaxes(np.swapaxes(im, 0, 2), 1, 2)
    im = torch.from_numpy(im)
    
    a = random.uniform(0, 2)
    b = random.uniform(-50, 50)
    
    out = linear_stretch(im, a, b)
    print(f"Parameters: a={a:.2f}, b={b:.2f}")
    print(out)


if __name__ == "__main__":
    main()
