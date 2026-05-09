"""GPT model with (Dense) Mixture of Experts Architecture."""
import math
from utils import Value, linear, softmax, rmsnorm

def gpt_moe(token_id, pos_id, keys, values, state_dict, n_layer, n_head, head_dim, heads_per_group, n_embd, n_experts, expert_sparsity):
    tok_emb = state_dict['wte'][token_id] # token embedding
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    # x = [t for t, p in zip(tok_emb, pos_emb)] # not adding PE
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
            k_h = [ki[hs_kv:hs_kv+head_dim] for ki in keys[li]]
            v_h = [vi[hs_kv:hs_kv+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        # 2) Gating
        x_residual = x
        x = rmsnorm(x)
        logits = linear(x, state_dict[f'layer{li}.moe_gate'])
        
        # sparisification
        hs = {} # hash map, value->gate_index
        for i, logit in enumerate(logits) : hs[logit] = i
        hs = dict(sorted(hs.items(), key=lambda x:x[0])) # sort by keys
        for i, index in enumerate(hs.values()) :
            if i<expert_sparsity : continue
            logits[index] = Value(-float('inf')) # masking all other expert logits

        gate_weights = softmax(logits)

        # 3) getting expert outputs
        expert_outputs = []
        for j in range(n_experts) :
            expert_x = linear(x, state_dict[f'layer{li}.moe_expert_{j+1}_fc1'])
            expert_x = [xi.relu() for xi in expert_x]
            expert_x = linear(expert_x, state_dict[f'layer{li}.moe_expert_{j+1}_fc2'])
            expert_outputs.append(expert_x)


        # combining the expert outputs according to the gate weights
        x = [sum(gate_weights[j] * expert_outputs[j][i] for j in range(n_experts)) for i in range(n_embd)]
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits