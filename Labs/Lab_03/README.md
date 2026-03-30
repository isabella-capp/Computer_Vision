# Residual Block 
A residual block is defined as:

$y=σ(F(x)+G(x))$

where:
- $\mathbf{x}$ and $\mathbf{y}$ represent the input and output tensors of the block, 
- $\sigma$ is the ReLU activation function,
- $\mathcal{F}$ is the residual function to be learned 
- $\mathcal{G}$ is a projection shortcut used to match dimensions between $\mathcal{F}(\mathbf{x})$ and $\mathbf{x}$.

Your code needs to define a ResidualBlock class (inherited from `nn.Module`) which implements a residual block. 

In your code, $\mathcal{F}$ will be implemented with two convolutional layers with a ReLU non-linearity between them, i.e. $\mathcal{F} = \texttt{conv}_2(\sigma(\texttt{conv}_1(\mathbf{x})))$. Batch normalization will also be adopted right after each convolution operation.

The constructor of the ResidualBlock class needs to take the following arguments as input:
- `inplanes`, the number of channels of $\mathbf{x}$;
- `planes`, the number of output channels of $\texttt{conv}_1$ and $\texttt{conv}_2$;
- `stride`, the stride of $\texttt{conv}_1$.

If the shapes of $\mathcal{F}(\mathbf{x})$ and $\mathbf{x}$ do not match (either because `inplanes != planes`, or because `stride > 1`) ResidualBlock also needs to apply a projection shortcut $\mathcal{G}$, composed of a convolutional layer with kernel size $1\times 1$, no bias, no padding and stride stride, followed by a batch normalization. Otherwise $\mathcal{G}$ is simply the identity function.

The `forward` method of `ResidualBlock` will take as input the input tensor $\mathbf{x}$ and return the output tensor $\mathbf{y}$, after performing all the operations of a Residual block.

Additional details: unless otherwise specified, convolutional layers must have $3 \times 3$ kernels, stride 1, padding 1 and no bias.