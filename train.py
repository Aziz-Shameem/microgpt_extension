"""Training script for GPT models with model selection """
import os
import math
import random
import argparse
from models import MODEL_REGISTRY
from utils import Value, softmax

def call_model(model_fn, token_id, pos_id, keys, values, state_dict, n_layer, n_head, 
               head_dim, heads_per_group, n_tokens, n_embd, block_size, kv_chunk_size, n_experts, expert_sparsity):
    """Wrapper to call model functions with the correct parameters."""
    # Get model name by checking function name
    model_name = model_fn.__name__.replace('gpt_', '')
    
    if model_name == 'baseline':
        return model_fn(token_id, pos_id, keys, values, state_dict, n_layer, n_head, 
                       head_dim, heads_per_group)
    elif model_name == 'rope':
        return model_fn(token_id, pos_id, keys, values, state_dict, n_layer, n_head, 
                       head_dim, heads_per_group, n_embd)
    elif model_name == 'xpos':
        return model_fn(token_id, pos_id, keys, values, state_dict, n_layer, n_head, 
                       head_dim, heads_per_group, n_embd)
    elif model_name == 'alibi':
        return model_fn(token_id, pos_id, keys, values, state_dict, n_layer, n_head, 
                       head_dim, heads_per_group)
    elif model_name == 't5_bias':
        return model_fn(token_id, pos_id, keys, values, state_dict, n_layer, n_head, 
                       head_dim, heads_per_group, block_size)
    elif model_name == 'flash':
        return model_fn(token_id, pos_id, keys, values, state_dict, n_layer, n_head, 
                       head_dim, heads_per_group, kv_chunk_size)
    elif model_name == 'mtp_naive':
        return model_fn(token_id, pos_id, keys, values, state_dict, n_layer, n_head, 
                       head_dim, heads_per_group, n_tokens)
    elif model_name == 'moe':
        return model_fn(token_id, pos_id, keys, values, state_dict, n_layer, n_head, 
                        head_dim, heads_per_group, n_embd, n_experts, expert_sparsity)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train GPT model with different positional encoding strategies')
    parser.add_argument('--model', type=str, default='baseline', 
                       choices=list(MODEL_REGISTRY.keys()),
                       help='Model architecture to use')
    parser.add_argument('--num-steps', type=int, default=500,
                       help='Number of training steps')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of samples to generate during inference')
    return parser.parse_args()

# Set random seed for reproducibility
random.seed(42)

if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
encoding = {ch:i for i, ch in enumerate(uchars)} 
decoding = {i:ch for i, ch in enumerate(uchars)}
vocab_size = len(uchars)
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

# iterations, put 0 for no bpe
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

# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension)
block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
n_head = 4      # number of attention heads
n_group = n_head # for GQA : n_group=1 implements MQA, n_group=n_head implements standard MHA
assert n_head % n_group == 0, "number of heads must be divisible by number of n_group"
heads_per_group = n_head // n_group
head_dim = n_embd // n_head # derived dimension of each head
kv_chunk_size = 2 # for FlashAttention - number of groups to process together for efficiency (e.g. 2 groups = 2*head_dim dims for k and v)
n_tokens = 3 # for multi-token prediction
n_experts = 3 # for MoE
expert_sparsity = 2 # number of experts each token is routed to
expert_sparsity = max(1, min(expert_sparsity, n_experts)) # clipping to [1, n_experts]
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {} # dictionary to hold all model parameters, organized by layer and function
params = [] # list to hold all parameters for easy access during optimization

# modularizing the weight initialization to be called after model selection, since different models have different parameters 
# (e.g. MTP has multiple lm_head's, MoE has gating and expert parameters, etc.)
def populate_weights(model_name):
    global state_dict, params

    state_dict = {
        'wte': matrix(vocab_size, n_embd),
        'wpe': matrix(block_size, n_embd),
        'lm_head': matrix(vocab_size, n_embd),
    }

    if model_name == 'mtp_naive':
        for i in range(2, n_tokens + 1):
            state_dict[f'lm_head{i}'] = matrix(vocab_size, n_embd)

    for i in range(n_layer):
        state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_group * head_dim)
        state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_group * head_dim)
        state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
        if model_name != 'moe' :
            state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
            state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

        if model_name == 't5_bias':
            state_dict[f'layer{i}.rel_pos_bias'] = matrix(n_head, 2 * block_size + 1)

        if model_name == 'moe':
            state_dict[f'layer{i}.moe_gate'] = matrix(n_experts, n_embd)
            for j in range(n_experts):
                state_dict[f'layer{i}.moe_expert_{j+1}_fc1'] = matrix(2 * n_embd, n_embd)
                state_dict[f'layer{i}.moe_expert_{j+1}_fc2'] = matrix(n_embd, 2 * n_embd)

    params = [p for mat in state_dict.values() for row in mat for p in row]
    print(f"num params: {len(params)}")

# Adam params
def adam_params() :
    learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
    m = [0.0] * len(params) # first moment buffer
    v = [0.0] * len(params) # second moment buffer
    return learning_rate, beta1, beta2, eps_adam, m, v

def main():
    args = parse_args()
    
    # Get the model function from registry
    model_fn = MODEL_REGISTRY[args.model]
    print(f"\nUsing model: {args.model}")
    print(f"Training for {args.num_steps} steps\n")

    # populating the state dict
    populate_weights(args.model)

    # Adam optimizer parameters
    learning_rate, beta1, beta2, eps_adam, m, v = adam_params()
    
    # Repeat in sequence
    num_steps = args.num_steps
    for step in range(num_steps):

        # Take single document and tokenize it
        doc = docs[step % len(docs)]
        tokens = encoded_docs[step % len(docs)] # use pre-encoded version from bpe
        n = min(block_size, len(tokens) - 1)

        # Forward the token sequence through the model, building up the computation graph all the way to the loss
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            target_ids = [target_id] + [tokens[pos_id + i] if pos_id + i < len(tokens) else None for i in range(2, n_tokens + 1)]

            # Call the model with the wrapper function
            logits = call_model(model_fn, token_id, pos_id, keys, values, state_dict, n_layer, n_head, 
                              head_dim, heads_per_group, n_tokens, n_embd, block_size, kv_chunk_size, n_experts, expert_sparsity)
            
            if isinstance(logits[0], list) :
                loss_t = 0.0
                probs = [softmax(logits[i]) for i in range(len(logits))] # compute probabilities for each token
                for i, (prob, target_id) in enumerate(zip(probs, target_ids)) :
                    if target_id is not None:
                        loss_t -= (1.0/(i+1))*prob[target_id].log()
            else :
                probs = softmax(logits)
                loss_t = -probs[target_id].log()
            losses.append(loss_t)
        loss = (1 / n) * sum(losses) # final average loss over the document sequence. 

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

    # Inference
    temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
    print(f"\n--- inference (model: {args.model}) ---")
    for sample_idx in range(args.num_samples):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        token_id = BOS
        sample = []
        for pos_id in range(block_size):
            logits = call_model(model_fn, token_id, pos_id, keys, values, state_dict, n_layer, n_head, 
                              head_dim, heads_per_group, n_tokens, n_embd, block_size, kv_chunk_size, n_experts, expert_sparsity)
            if isinstance(logits[0], list) : logits = logits[0] # using only the next token prediction
            probs = softmax([l / temperature for l in logits])
            token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if token_id == BOS:
                break
            sample.append(decoding[token_id])
        print(f"sample {sample_idx+1:2d}: {''.join(sample)}")

if __name__ == '__main__':
    main()
