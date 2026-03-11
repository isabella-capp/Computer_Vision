import random
import torch


def max_pooling_2d(input, kH, kW, s):
    """Apply 2D max pooling to input tensor.
    
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
    
    for idx_n in range(n):
        image = input[idx_n]
        for row in range(oH):
            for col in range(oW):    
                for i in range(iC):
                    submatrix = image[i, row*s:row*s+kH, col*s:col*s+kW]   
                    out[idx_n][i][row][col] = torch.max(submatrix)
    
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
    
    out = max_pooling_2d(input, kH, kW, s)
    print(out)


if __name__ == "__main__":
    main()