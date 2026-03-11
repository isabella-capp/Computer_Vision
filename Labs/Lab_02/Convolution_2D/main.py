import random
import torch


def convolution_2d(input, kernel):
    """Apply 2D convolution to input tensor.
    
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
    
    for idx_n in range(n):
        image = input[idx_n]
        for row in range(oH):
            for col in range(oW):
                submatrix = image[:,row:row+kH, col:col+kW]            
                for k in range(oC):
                    out[idx_n][k][row][col] = torch.sum(submatrix * kernel[k])
    
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
    
    out = convolution_2d(input, kernel)
    print(out)


if __name__ == "__main__":
    main()