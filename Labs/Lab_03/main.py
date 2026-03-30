import torch
import torch.nn as nn
import torch.nn.functional as f

class ResidualBlock(nn.Module):

    def __init__(self, inplanes, planes, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=planes)

        if stride > 1 or inplanes != planes:
            self.conv_g = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=stride, bias=False )
            self.bn_g = nn.BatchNorm2d(num_features=planes)
        else:
            self.conv_g = None
            self.bn_g = None

    def forward(self, X):
        F = self.bn2(self.conv2(f.relu(self.bn1(self.conv1(X)))))
        
        if self.conv_g is not None and self.bn_g is not None:
            G = self.bn_g(self.conv_g(X))
        else:
            G = X

        return f.relu(F+G)
    

if __name__ == "__main__":
    block = ResidualBlock(inplanes=256, planes=256, stride=2)
    x = torch.randn(1, 256, 56, 56)
    out = block(x)
    print(out.shape)
    



