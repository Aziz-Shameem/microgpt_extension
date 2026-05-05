"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.
@karpathy
"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
from utils import *    # Value class for autograd and other utilities
random.seed(42) # Let there be order among chaos

# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
# docs = ['\n' + doc + '\n' for doc in docs] # add special BOS token to mark the beginning and end of each document
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
encoding = {ch:i for i, ch in enumerate(uchars)} 
decoding = {i:ch for i, ch in enumerate(uchars)}
# BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

## byte-pair encoding (BPE) : iteratively find the most common pair of tokens and merge them into a new token
def run_bpe(bigrams, encoding, decoding, encoded_docs) :
    top_combination, _ = max(bigrams.items(), key=lambda item: item[1])
    top_combination_str = decoding[top_combination[0]] + decoding[top_combination[1]]
    new_token_id = len(encoding)
    new_encoding = encoding.copy()
    new_encoding[top_combination_str] = new_token_id
    new_decoding = decoding.copy()
    new_decoding[new_token_id] = top_combination_str

    # Re-encode documents using INTEGER token sequences, not strings
    new_encoded_docs = []
    old_bigram = top_combination
    for encoded_doc in encoded_docs :
        new_doc = []
        i = 0
        while i < len(encoded_doc):
            if i < len(encoded_doc)-1 and (encoded_doc[i], encoded_doc[i+1]) == old_bigram:
                new_doc.append(new_token_id)
                i += 2
            else:
                new_doc.append(encoded_doc[i])
                i += 1
        new_encoded_docs.append(new_doc)

    # Recompute bigrams from new encoded docs
    new_bigrams = {}
    for encoded_doc in new_encoded_docs :
        for a, b in zip(encoded_doc, encoded_doc[1:]):
            new_bigrams[(a,b)] = new_bigrams.get((a,b), 0) + 1
    new_bigrams = dict(sorted(new_bigrams.items(), key=lambda x: x[1], reverse=True))

    print(f'Vocabulary size : {len(new_encoding)}')
    return new_bigrams, new_encoding, new_decoding, new_encoded_docs

# initial run
bigrams = {}
encoded_docs = [[encoding[ch] for ch in doc] for doc in docs]
for encoded_doc in encoded_docs :
    for a,b in zip(encoded_doc, encoded_doc[1:]):
        bigrams[(a,b)] = bigrams.get((a,b), 0) + 1
bigrams = {k:v for k,v in sorted(bigrams.items(), key=lambda item: item[1], reverse=True)}

# iteraions
n_iter = 10
for _ in range(n_iter) :
    bigrams, encoding, decoding, encoded_docs = run_bpe(bigrams, encoding, decoding, encoded_docs)

# adding BOS token as a newline character in BPE encoding
encoding['\n'] = BOS = len(encoding)
decoding[len(decoding)] = '\n'
encoded_docs = [[encoding['\n']]+x+[encoding['\n']] for x in encoded_docs] # add BOS token at the beginning and end of each document

# Update vocab_size after BPE merges
vocab_size = len(encoding)
print(f"vocab size after BPE: {vocab_size}")
print(f'Encodings map : {encoding}')

# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension)
block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
n_head = 4      # number of attention heads
n_group = n_head # for GQA : n_group=1 implements MQA, n_group=n_head implements standard MHA
assert n_head % n_group == 0, "number of heads must be divisible by number of n_group"
heads_per_group = n_head // n_group
head_dim = n_embd // n_head # derived dimension of each head
kv_chunk_size = 2 # number of groups to process together for efficiency (e.g. 2 groups = 2*head_dim dims for k and v)
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_group*head_dim)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_group*head_dim)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
    state_dict[f'layer{i}.rel_pos_bias'] = matrix(n_head, 2*block_size+1) # T5-style relative positional bias: one value per head per relative distance
params = [p for mat in state_dict.values() for row in mat for p in row] # flatten params into a single list[Value]
print(f"num params: {len(params)}")

# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

## build rotation matrix for 2D rotary embeddings - RoPE
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

# rotation matrix for a given position and base frequency - RoPE
R_matrix = lambda m, theta: [
    [Value(m * theta).cos(), -Value(m * theta).sin()],
    [Value(m * theta).sin(),  Value(m * theta).cos()]
]

# matrix multiplication - equivalent to '@' in numpy
def matrix_mul(A, B):
    return [
        [
            sum(A[i][k] * B[k][j] for k in range(len(B)))
            for j in range(len(B[0]))
        ]
        for i in range(len(A))
    ]

def gpt(token_id, pos_id, keys, values):
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
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

def gpt_with_t5_bias(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id] # token embedding
    x = tok_emb # not adding absolute PE
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

            # Add T5-style relative positional bias: for each key position, add learned bias based on relative distance
            for t in range(len(k_h)):
                rel_dist = t - pos_id  # relative distance (can be negative)
                rel_dist_idx = rel_dist + block_size  # offset for indexing (to handle negative indices)
                if 0 <= rel_dist_idx < 2*block_size+1:
                    attn_logits[t] = attn_logits[t] + state_dict[f'layer{li}.rel_pos_bias'][h][rel_dist_idx]

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

def gpt_with_alibi(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id] # token embedding
    x = tok_emb # not adding absolute PE
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

def gpt_with_rope(token_id, pos_id, keys, values):
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

def gpt_flash_attention(token_id, pos_id, keys, values):
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

# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # first moment buffer
v = [0.0] * len(params) # second moment buffer

# Repeat in sequence
num_steps = 500 # number of training steps
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    # tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    tokens = encoded_docs[step % len(docs)] # use pre-encoded version from bpe
    n = min(block_size, len(tokens) - 1)

    # Forward the token sequence through the model, building up the computation graph all the way to the loss
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt_with_rope(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses) # final average loss over the document sequence. May yours be low.

    # Backward the loss, calculating the gradients with respect to all model parameters
    loss.backward()

    # Adam optimizer update: update the model parameters based on the corresponding gradients
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')
print()
print(f'Minimum loss: {min(losses).data:.4f}')

# Inference: may the model babble back to us
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt_with_rope(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(decoding[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")

# Inference with BPE tokens: generate new tokens in continuation
# temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
# print("\n--- inference (new, hallucinated names with BPE) ---")
# for sample_idx in range(20):
#     keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
#     # Start with a newline token (BOS marker in BPE encoding)
#     sample_tokens = [encoding['\n']]
    
#     # Generate tokens continuously for n iterations
#     n_generate = 20  # number of tokens to generate
#     for gen_step in range(n_generate):
#         token_id = sample_tokens[-1]
#         # Use modulo to wrap position within block_size
#         pos_id = gen_step % block_size
#         logits = gpt(token_id, pos_id, keys, values)
#         probs = softmax([l / temperature for l in logits])
#         next_token = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
#         sample_tokens.append(next_token)
        
#         # Maintain context window: keep only last block_size tokens in KV cache
#         if len(keys[0]) > block_size:
#             for li in range(n_layer):
#                 keys[li] = keys[li][-block_size:]
#                 values[li] = values[li][-block_size:]
    
#     # Decode all tokens to text
#     sample_text = ''
#     for token_id in sample_tokens:
#         token_str = decoding.get(token_id, '?')
#         sample_text += token_str
    
#     # Remove the leading/trailing newlines for display
#     sample_text = sample_text.strip()
#     print(f"sample {sample_idx+1:2d}: {sample_text}")
