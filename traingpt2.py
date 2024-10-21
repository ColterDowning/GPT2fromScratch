
# The goal of this exercise is to load in the hugging face implementation of gpt2
# (because the openAI version is written using Tensorflow not PyTorch), and then
# create our own from-scratch and trained version of the gpt2 model to hopefully
# surpass the performance of the actual gpt2 model. Let's jump in!

from dataclasses import dataclass # For decorators
import math
import tiktoken
import torch # Using PyTorch
import torch.nn as nn
from torch.nn import functional as F

#--------------------------------------------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    # All of this is equivalent to our implementation from the original gpt project, this is just 
    # more efficient for PyTorch

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # 'bias' is written here, but it represents the mask. Just using to follow OpenAI/HF naming
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # nh is number of heads, hs is head size, C is number of channels = nh * hs
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        # attention that creates the (T, T) matrix for all queries and keys
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Masks future tokens so the current token can only learn from previous tokens
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) 
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1,2).contiguous().view(B, T, C) # reassemble the head outputs
        # output projection
        y = self.c_proj(y)
        return y




class MLP(nn.Module):

    # The inner layer dimensionality of the feed forward is 4x, so we grow our input
    # dimension (n_embd) by 4, and then scale it back down by 4 at the projection
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module): #sub-class of nn.Module
    # Implements each of the transformer blocks

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x): # Layer norm, then attention, then another layer norm, then mlp
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass # Decorator that tells Python to automatically generate special methods for the
# GPT config class like __init__, __repr__, __eq__, __hash__, etc. 
class GPTConfig:
    block_size: int = 1024 # Size of the context window, used to create positional
    # embeddings and sets the upper limit on how many tokens can attend to 
    # eachother.
    vocab_size: int = 50257 # number of unique tokens that the model knows
    n_layer: int = 12 # Depth of the model (number of layers)
    n_head: int = 12 # Number of attention heads
    n_embd: int = 768 # Dimensionality of the embedding space

class GPT(nn.Module): # Sub-class of the nn.Module, meaning GPT gets the methods and 
    # attributes of nn.Module

    # GPT2 is a decoder only architecture, so there is not cross-attention from the encoder.
    # There are a couple other differences between GPT2 and the original transformer
    # architecture from the "Attention is all you need" paper.
    # 1. Layer normalization was shuffled around to be at the beginning of each block
    # 2. A final layer norm was added after the final self-attention block

    # Let's create some of the sub modules following the hugging face transformer schema 

    def __init__(self, config): # Initializes an instance of the GPT class. 
        super().__init__() # Inherits from nn.Module (calls the constructor of nn.Module)
        self.config = config

        self.transformer = nn.ModuleDict(dict( # Main module. Is a dict so you can index
        # into the submodules using a key

            # nn.Embedding is a fancy wrapper around a tensor array
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Weights of token embeddings (output embeddings in Transformer paper)
            wpe = nn.Embedding(config.block_size, config.n_embd), # Weights of position embeddings (positional embeddings in Transformer paper)
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # hidden layers (all blocks in grey in Transformer paper)
            ln_f = nn.LayerNorm(config.n_embd), # Final layer norm added in gpt2
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # Final projection
        # from embedding space to our vocabulary (linear block in Transformer paper)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        #init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'): # scale to control std dev
                std *= (2 * self.config.n_layer) ** -0.5 # mlp and attn forward for each block, so 2x
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None): # The inputs is our indices, our tokens
        # idx is of shape (B,T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape T
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer norm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # The cross entropy function does not like multidimensional tensors, so we have to flatten it out to just 2 dims
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type): # Operates on the class itself, not an instance of the class (although it could if we want)
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head, and n_embd are determined from model_type
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),  # 124M parameters
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M parameters
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280), # 774M parameters
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600), # 1558M parameters
        }[model_type]
        config_args['vocab_size'] = 50257 # 50,000 BPE merges, 256 byte tokens, 1 <|endoftext|> token
        config_args['block_size'] = 1024 # always 1024 for gpt

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict() # Creating our state dictionary for our model 
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask/buffer

        # Initialize a huggingface tranformer model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # discard this mask/buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same thing, ignore the bias
        # For some reason, the hugging face weights are transposed from what PyTorch wants. Probably some bi-product
        # of taking these from Tensorflow?
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
#----------------------------------------------------------------------------------------------------

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        #state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch is out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

#----------------------------------------------------------------------------------------------------

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(f"using device: {device}")
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B=4, T=32)

#model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model.to(device)

#optimize! AdamW is a faster way to optimize compared to stochastic gradient descent
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad() # Make sure to always 0 the gradients first!
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")


import sys; sys.exit(0)

# prefix tokens
model.eval()
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5,8)
x = tokens.to(device)
# x is now our idx that we put into forward to generate new tokens!

# generate! right now x is (B, T) where B = 5 and T = 8
# set seed to 42 because it's the answer to the universe (and the seed we have been using all this time)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length: # stop generating at the defined max token length
    with torch.no_grad(): # not going to call .backward() on any of the following code, so don't cache anything
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position (inefficient, but it works)
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        # topk sampling keeps the top 50 probabilties and clamps the rest to 0, then re-normalizes
        # helps keep the model on track and prevent blabbering lol
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)