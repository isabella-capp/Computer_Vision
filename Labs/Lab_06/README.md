# 2D Transpose Convolution

Your code will take an input tensor input with shape `(n, iC, H, W)`, a kernel kernel with shape `(iC, oC, kH, kW)` and a stride `s`.

It needs then to apply a 2D Transpose convolution over `input`, using `kernel` as kernel tensor, using a stride of `s` on both axes, no dilation, no grouping, and no padding, and store the result in `out`.

`input` and `kernel` are `torch.Tensor` with dtype `torch.float32`. `s` is an integer.