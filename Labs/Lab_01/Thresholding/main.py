import random
import numpy as np
from skimage import data
from skimage.transform import resize
import torch


def thresholding(im, val):
    """Apply binary thresholding to image.
    
    Args:
        im: Input image tensor
        val: Threshold value
    
    Returns:
        Binary thresholded image
    """
    return torch.where(im > val, 255, 0).to(torch.uint8)


def main():
    im = data.camera()
    im = resize(im, (im.shape[0] // 2, im.shape[1] // 2), mode='reflect', preserve_range=True, anti_aliasing=True)
    im = np.asarray(im, dtype=np.uint8)
    im = torch.from_numpy(im)
    val = random.randint(0, 255)
    
    out = thresholding(im, val)
    print(f"Threshold value: {val}")
    print(out)


if __name__ == "__main__":
    main()