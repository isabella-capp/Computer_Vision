import torch

def scaled_dot_product_attention(query, keys, values):                        
    """Calculate the scaled dot product attention.
    
    Args:
        query: Query tensor of shape (b, T, d_k)
        key: Key tensor of shape (b, T, d_k)
        value: Value tensor of shape (b, T, d_v)
    
    Returns:
        Output tensor after applying attention
    """
    b, T, d_k = query.shape

    argument = query @ torch.transpose(keys, -2, -1) / d_k**0.5

    return torch.softmax(input=argument, dim=-1) @ values

if __name__ == "__main__":
    batch_size = 2
    seq_length = 3
    d_k = 4  
    d_v = 5 

    q = torch.randn(batch_size, seq_length, d_k)
    k = torch.randn(batch_size, seq_length, d_k)
    v = torch.randn(batch_size, seq_length, d_v)

    # 3. Run the function
    output = scaled_dot_product_attention(q, k, v)

    # 4. Verify the results
    print(f"Query shape:  {q.shape}")
    print(f"Keys shape:   {k.shape}")
    print(f"Values shape: {v.shape}")
    print("-" * 30)
    print(f"Output shape: {output.shape} -> Expected: ({batch_size}, {seq_length}, {d_v})")
    print("\nOutput Tensor:")
    print(output)