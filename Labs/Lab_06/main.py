import random
import torch

def _2D_transpose_convolution(input, kernel, s):
    """
    input: (n, iC, H, W)
    kernel: (iC, oC, kH, kW)
    s: int

    return:
        out: (n, oC, oH, oW)
    """
    
    N, iC, H, W = input.shape
    _, oC, kH, kW = kernel.shape
    
    oH = (H-1) * s + kH 
    oW = (W-1) * s + kW 

    out = torch.zeros((N, oC, oH, oW), dtype=torch.float32)

    input = input.view(N, iC, H, W, 1, 1, 1)
    kernel = kernel.view(1, iC, 1, 1, oC, kH, kW)    

    matrix_mult = (input * kernel).sum(dim=1)
    for row in range(H):
        for col in range(W):
            out[:,:,row*s:row*s+kH, col*s:col*s+kW]  += matrix_mult[:,row,col,:, :,:]
    
    return out


def main():
    n = random.randint(2, 6)
    iC = random.randint(2, 6)
    oC = random.randint(2, 6)
    H = random.randint(10, 20)
    W = random.randint(10, 20)
    kH = random.randint(2, 6)
    kW = random.randint(2, 6)
    s = random.randint(2, 6)

    input = torch.rand(n, iC, H, W)
    kernel = torch.rand(iC, oC, kH, kW)
    out = _2D_transpose_convolution(input, kernel, s)
    print(out)
   

if __name__ == "__main__":
    main()