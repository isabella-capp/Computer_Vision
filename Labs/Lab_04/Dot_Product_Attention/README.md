# Scaled Dot-Product Attention Operator

The core of the multi-head self-attention mechanism is the scaled dot-product attention. Before building the full module with learnable projections, your first task is to implement this core parameter-free operation.

Your code needs to define a function scaled_dot_product_attention which computes the attention weights
and the aggregated values. The function needs to take the following arguments with length T as input:
- `q`: the query tensor `(b, T, dk)`
- `k`: the key tensor `(b, T, dk)`
- `v`: the value tensor `(b, T, dv)`

In your code, the operator should be implemented according to the standard mathematical formula.
The execution must entail the following steps:
- Compute the dot product between queries `q` and keys `k` by transposing the appropriate dimensions of the key tensor.
- Scale the resulting scores by dividing them by the square root of the head dimensionality `dk`.
- Apply the softmax activation function along the last dimension to obtain the normalized attention weights.
- Multiply the attention weights by the values `v` to get the final output.

The function must return the resulting output tensor