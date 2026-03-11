import random
import torch


def max_pooling_2d_optimized(input, kH, kW, s):
    """Apply 2D max pooling to input tensor (optimized version).
    
    Args:
        input: Input tensor of shape (n, iC, H, W)
        kH: Kernel height
        kW: Kernel width
        s: Stride
    
    Returns:
        Output tensor after max pooling
    """
    n, iC, H, W = input.shape
    oH = ((H-(kH-1)-1)//s)+1
    oW = ((W-(kW-1)-1)//s)+1
    
    out = torch.zeros(n, iC, oH, oW)
    
    for row in range(oH):
        for col in range(oW):    
            submatrix = input[:,:,row*s:row*s+kH, col*s:col*s+kW]   
            out[:,:,row,col] = torch.amax(submatrix, dim=(2, 3))
    
    return out


def main():
    n = random.randint(2, 6)
    iC = random.randint(2, 6)
    H = random.randint(10, 20)
    W = random.randint(10, 20)
    kH = random.randint(2, 5)
    kW = random.randint(2, 5)
    s = random.randint(2, 3)
    input = torch.rand((n, iC, H, W), dtype=torch.float32)
    
    out = max_pooling_2d_optimized(input, kH, kW, s)
    print(out)


if __name__ == "__main__":
    main()