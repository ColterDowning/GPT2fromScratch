
# The goal of this exercise is to load in the hugging face implementation of gpt2
# (because the openAI version is written using Tensorflow not PyTorch), and then
# create our own from-scratch and trained version of the gpt2 model to hopefully
# surpass the performance of the actual gpt2 model. Let's jump in!

from dataclasses import dataclass # For decorators
import math
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
        y = y.transpose(1,2).continuous().view(B, T, C) # reassemble the head outputs
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
        x += self.attn(self.ln_1(x))
        x += self.mlp(self.ln_2(x))
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