import random
import torch


def convolution_2d_optimized(input, kernel):
    """Apply 2D convolution to input tensor (optimized version).
    
    Args:
        input: Input tensor of shape (n, iC, H, W)
        kernel: Kernel tensor of shape (oC, iC, kH, kW)
    
    Returns:
        Output tensor after convolution
    """
    n, iC, H, W = input.shape
    oC, _, kH, kW = kernel.shape
    
    oH = H - (kH-1) 
    oW = W - (kW-1) 
    
    out = torch.zeros(n, oC, oH, oW)
    
    for row in range(oH):
        for col in range(oW):
            submatrix = input[:,:,row:row+kH, col:col+kW]  
            out[:,:,row,col] = torch.sum(submatrix.unsqueeze(1) * kernel.unsqueeze(0), dim=(2,3,4))
    
    return out


def main():
    n = random.randint(2, 6)
    iC = random.randint(2, 6)
    oC = random.randint(2, 6)
    H = random.randint(10, 20)
    W = random.randint(10, 20)
    kH = random.randint(2, 6)
    kW = random.randint(2, 6)
    
    input = torch.rand(n, iC, H, W, dtype=torch.float32)
    kernel = torch.rand(oC, iC, kH, kW, dtype=torch.float32)
    
    out = convolution_2d_optimized(input, kernel)
    print(out)


if __name__ == "__main__":
    main()          