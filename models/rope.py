"""GPT model with RoPE (Rotary Position Embeddings)."""
from utils import linear, softmax, rmsnorm, Value

def build_rotation_matrix(blocks, dim):
    """
    blocks: list of 2x2 matrices (as nested lists)
    dim: total dimension
    """
    # initialize dim x dim zero matrix
    R = [[Value(0.0) for _ in range(dim)] for _ in range(dim)]

    idx = 0
    for block in blocks:
        R[idx][idx]       = block[0][0]
        R[idx][idx + 1]   = block[0][1]
        R[idx + 1][idx]   = block[1][0]
        R[idx + 1][idx+1] = block[1][1]
        idx += 2

    # if odd dimension → last diagonal entry = 1
    if dim % 2 == 1:
        R[-1][-1] = 1.0

    return R

def R_matrix(m, theta):
    return [
        [Value(m * theta).cos(), -Value(m * theta).sin()],
        [Value(m * theta).sin(),  Value(m * theta).cos()]
    ]

def matrix_mul(A, B):
    return [
        [
            sum(A[i][k] * B[k][j] for k in range(len(B)))
            for j in range(len(B[0]))
        ]
        for i in range(len(A))
    ]

def gpt_rope(token_id, pos_id, keys, values, state_dict, n_layer, n_head, head_dim, heads_per_group, n_embd):
    tok_emb = state_dict['wte'][token_id] # token embedding

    rotation_matrices = [R_matrix(pos_id, 10000**(-2*block_id/head_dim)) for block_id in range(head_dim // 2)] # precompute rotation matrices for each block
    rotation_matrices = rotation_matrices * n_head # repeat for each head - apply rope to each head independently
    rotation_matrix_full = build_rotation_matrix(rotation_matrices, n_embd)

    x = tok_emb 
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        q_with_position = matrix_mul(rotation_matrix_full, [[qi] for qi in q]) # apply RoPE rotation to the query
        q_with_position = [x[0] for x in q_with_position] 
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        k_with_position = matrix_mul(rotation_matrix_full, [[ki] for ki in k]) # apply RoPE rotation to the key
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append([x[0] for x in k_with_position])
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            hs_kv = (h // heads_per_group) * head_dim
            q_h = q_with_position[hs:hs+head_dim]
            k_h = [ki[hs_kv:hs_kv+head_dim] for ki in keys[li]]
            v_h = [vi[hs_kv:hs_kv+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits



