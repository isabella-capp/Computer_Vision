# Otsu Thresholding

Given an input grayscale image `im` (a `torch.Tensor` with shape `(H, W)` and dtype `torch.uint8`), write a code which computes the Otsu threshold for `im` and stores the result in `out`.

>**Notice**: beware of how the threshold is defined in the Otsu formulas. Your output should be compliant with our first definition of threshold (see slides).

### Problem Setup
```python
import random
import numpy as np
from skimage import data
from skimage.transform import resize
import torch

im = data.camera()
im = resize(im, (im.shape[0] // 2, im.shape[1] // 2), mode='reflect', preserve_range=True, anti_aliasing=True).astype(np.uint8)
im = torch.from_numpy(im)
```