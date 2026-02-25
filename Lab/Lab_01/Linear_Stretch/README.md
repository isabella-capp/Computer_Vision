# Linear stretch

Your code will take as input a color image `im` (a `torch.Tensor` with dtype `torch.uint8` and rank 3) and two scalars `a` and `b`. 

It must apply a **pixel-wise linear transformation** (every pixel `p` is transformed to `a⋅p+b`). The code should produce a new image `out` with the same shape and dtype.

`a` and `b` can be either ints or floats. Be careful to: compute the exact result, round to nearest integer and then clip between 0 and 255.

### Problem Setup
```python
import random
import numpy as np
import torch
from skimage import data
from skimage.transform import resize

im = data.coffee()
im = resize(im, (im.shape[0] // 8, im.shape[1] // 8), mode='reflect', preserve_range=True, anti_aliasing=True).astype(np.uint8)
im = np.swapaxes(np.swapaxes(im, 0, 2), 1, 2)
im = torch.from_numpy(im)

a = random.uniform(0,2)
b = random.uniform(-50,50)
```