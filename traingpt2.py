
# The goal of this exercise is to load in the hugging face implementation of gpt2
# (because the openAI version is written using Tensorflow not PyTorch), and then
# create our own from-scratch and trained version of the gpt2 model to hopefully
# surpass the performance of the actual gpt2 model. Let's jump in!

from dataclasses import dataclass # For decorators
import os
import math
import time 
import tiktoken
import inspect
import torch # Using PyTorch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

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
        #att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Masks future tokens so the current token can only learn from previous tokens
        #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) 
        #att = F.softmax(att, dim=-1)
        #y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Instead of the 4 lines above computing our attention matrix, let's use flash attention
        # Flash attention is a clever alogorithm that never materializes the matrix values
        # on the GPU memory. Going back and forth between memory is the most time-consuming part,
        # so this saves us a bunch of time
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # Final projection aka classifier
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
    

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # Remember, we add weight decays to parameters to 'pull' down their weights so the
        # optimizations use more of the weights and distribute across the tokens more evenly.
        # We are not allowing any individual weight to be too large.

        # Here we will also use a fused Adam optimizer. This compiles the optimizer before runtime 
        # so all computations can be done in the GPU tensor cores without being materialized in HBM
        # memory.

        # start with all candidate parameters that require grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise not.
        # i.e all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params) # numel() returns number of individual weights in parameter p
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
#----------------------------------------------------------------------------------------------------

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
            train_val_cutoff = int(len(text) * 0.8)
            train_split_text = text[:train_val_cutoff]
            val_split_text = text[train_val_cutoff:]
        enc = tiktoken.get_encoding('gpt2')
        if split == 'train':
            tokens = enc.encode(train_split_text)
        elif split == 'val':
            tokens = enc.encode(val_split_text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens as a {split} split")

        #state
        self.reset()

    def reset(self):
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch is out of bounds, reset
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y

#----------------------------------------------------------------------------------------------------
# simple launch:
# python traingpt2.py
# DDP launch for 8 GPUS, for example
# torchrun --standalone --nproc_per_node=8 traingpt2.py


# Run the training loop
from torch.distributed import init_process_group, destroy_process_group

# Setup DDP (Distributed Data Parallel)
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # Is this a ddp run?
if ddp:
    # ddp requires CUDA
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK']) # identifier for each processes, like a label
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # rank of GPU on a single node
    ddp_world_size = int(os.environ['WORLD_SIZE']) # total number of processes running
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging and checkpointing
else:
    # vanilla non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(f"using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 32768 # 2**15 tokens. GPT2 has a batch size ~ 500k, but my GPU is smol
B = 4 # micro batch size
T = 256 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')
torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
#model = torch.compile(model) # use if your GPU CUDA compatability >= 7.0 
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it): # cosine decay learning scheduler =)
    # 1. linear warmup for warmup_iters steps
    if it < warmup_steps: 
        return max_lr * (it+1) / warmup_steps
    # 2. if it > lr_decay_iters, return min learning rate
    if it > max_steps:
         return min_lr
    # 3. In between, use the cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

#optimize! AdamW is a faster way to optimize compared to stochastic gradient descent
#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()

    # once in a while evaluate our validation loss
    if step % 5 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}") 

    # once in a while generate from the model
    if step > 0 and step % 5 == 0:
        model.eval()
        num_return_sequences = 1
        max_length = 32
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad(): # not going to call .backward() on any of the following code, so don't cache anything
                logits, loss = model(xgen) # (B, T, vocab_size)
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
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)

        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")


    # training loop
    model.train()
    optimizer.zero_grad() # Make sure to always 0 the gradients first!
    #with torch.autocast(device_type=device, dtype=torch.bfloat16): # If your GPU supports bfloats
    loss_accum = 0.0
    # Since my poor little GPU can't handle a batch size of 0.5M, we can simulate that batch size with
    # gradient accumulation. This is done by doing forward and backward passes grad_accum_steps
    # times, and then update the gradient afterwards.
    for microstep in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        loss = loss / grad_accum_steps # need to normalize over the accumulation
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (microstep == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # creates the average loss_accum across all ranks and deposits
        # on all ranks. After this call, all ranks will have the same loss_accum value 
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clipping the gradient norm to prevent 'shocks' from
    # unlucky batches. This sets an upper bound at 1.
    
    # Determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # make sure we wait for the scheduled work to actually finish
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f} sec | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()

import sys; sys.exit(0)

