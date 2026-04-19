# ROI Pooling
Implement a ROI Pooling operator. Your code will be given the following variables:

- `input`, a mini-batch of feature maps (a `torch.Tensor` with shape `(n, C, H, W)` and `dtype torch.float32`)
- `boxes`, a list of bounding box coordinates on which you need to perform the ROI Pooling. `boxes` will be a list of `(L,4)` `torch.Tensor` with `dtype torch.float32`, where `boxes[i]` will refer to the i-th element of the batch, and contain L coordinates in the format `(y1, x1, y2, x2)`
- a tuple of integers `output_size`, containing the number of cells over which pooling is performed, in the format `(heigth, width)`


The code should produce an output `torch.Tensor` out with `dtype torch.float32` and shape `(n, L, C, output_size[0], output_size[1])`.