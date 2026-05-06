"""GPT model with Flash Attention optimization."""
import math
from utils import linear, softmax, rmsnorm, Value

def gpt_flash(token_id, pos_id, keys, values, state_dict, n_layer, n_head, head_dim, heads_per_group, kv_chunk_size):
    tok_emb = state_dict['wte'][token_id] # token embedding
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

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
            l = Value(0.0)
            max_so_far = -math.inf
            head_out = [Value(0.0)] * head_dim
            for kv_ind in range(0, len(keys[li]), kv_chunk_size):
                k_h_chunk = [ki[hs_kv:hs_kv+head_dim] for ki in keys[li][kv_ind:kv_ind+kv_chunk_size]]
                v_h_chunk = [vi[hs_kv:hs_kv+head_dim] for vi in values[li][kv_ind:kv_ind+kv_chunk_size]]
                attn_logits = [sum(q_h[j] * k_h_chunk[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h_chunk))]
                
                # Find max in this chunk for numerical stability
                max_chunk = max([logit.data for logit in attn_logits]) if attn_logits else -math.inf
                
                # If we have a new max, rescale previous results
                if max_chunk > max_so_far:
                    if max_so_far != -math.inf:
                        scale_old = math.exp(max_so_far - max_chunk)
                        head_out = [x * scale_old for x in head_out]
                        l = l * scale_old
                    max_so_far = max_chunk
                
                # Compute softmax with numerical stability
                attn_logits_exp = [(logit - max_so_far).exp() for logit in attn_logits]
                attn_logits_exp_sum = sum(attn_logits_exp)
                l += attn_logits_exp_sum

                head_out_chunk = [sum(attn_logits_exp[t] * v_h_chunk[t][j] for t in range(len(v_h_chunk))) for j in range(head_dim)]
                head_out = [a+b for a, b in zip(head_out, head_out_chunk)]
            x_attn.extend([x/l for x in head_out])
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
