# Color Histogram

Your code will take as input a color image `im` (a `torch.Tensor` with dtype torch.uint8 and shape `(3, H, W)`) and an integer nbin. It should compute a normalized color histogram of the image, quantized with `nbin` bins on each color plane.

The output should be a `torch.Tensor` with shape `(3*nbin, )`, containing the concatenation of the histograms computed on each color plane (in the same order of the input tensor).

The output should be `L1`-normalized (i.e. all bins of the final histogram should sum up to 1).

Quantization strategy: a pixel should go in the bin with index `b` iif: `pixel*nbin // 256 == b`