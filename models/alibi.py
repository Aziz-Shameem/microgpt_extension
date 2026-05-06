"""GPT model with ALiBi (Attention with Linear Biases)."""
from utils import linear, softmax, rmsnorm

def gpt_alibi(token_id, pos_id, keys, values, state_dict, n_layer, n_head, head_dim, heads_per_group):
    tok_emb = state_dict['wte'][token_id] # token embedding
    x = tok_emb 
    x = rmsnorm(x) 

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            hs_kv = (h // heads_per_group) * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs_kv:hs_kv+head_dim] for ki in keys[li]]
            v_h = [vi[hs_kv:hs_kv+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]

            # add ALiBi style penalty based on absolute distance (pos_id) and learned slope per head
            param = 2**(-8*h/n_head)
            attn_logits = [x - param*(len(k_h)-y-1) for x,y in zip(attn_logits, range(len(k_h)))]

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
