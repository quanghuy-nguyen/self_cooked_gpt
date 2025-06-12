import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2Tokenizer
from dataclasses import dataclass


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class GPTConfig:
    vocab_size: int = 50257 # GPT2 vocab size
    embed_size: int = 64
    num_heads: int = 1
    num_layers: int = 1
    head_size: int = embed_size//num_heads
    batch_size: int = 4
    block_size: int = 128
    max_len: int = 128
    dropout: float = 0.3
    bias: bool = False

class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout_value = config.dropout
        self.query = nn.Linear(config.embed_size, config.head_size, bias=config.bias)
        self.key = nn.Linear(config.embed_size, config.head_size, bias=config.bias)
        self.value = nn.Linear(config.embed_size, config.head_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.register_buffer('tril', torch.tril(torch.ones((config.block_size, config.block_size))))

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_value if self.training else 0, is_causal=True)
        else:

            causal_mask = self.tril[:T, :T].unsqueeze(0).expand(B, -1, -1)

            y = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 
            y = y.masked_fill(causal_mask==0, float('-inf'))

            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1).expand(B, T, T)
                y = y.masked_fill(attention_mask==0, float('-inf'))

            y = F.softmax(y, dim=-1)
            y = self.dropout(y)
        
            y = y @ v
        
        return y
    

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.num_heads)])
        self.linear = nn.Linear(config.head_size*config.num_heads, config.embed_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask):
        x = torch.cat([head(x, attention_mask) for head in self.heads], dim=-1)
        x = self.linear(x)
        out = self.dropout(x)
        return out
    

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embed_size, config.embed_size*4, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.embed_size*4, config.embed_size, bias=config.bias),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.multi_heads = MultiHeadAttention(config)
        self.ffw = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.embed_size)
        self.ln2 = nn.LayerNorm(config.embed_size)
        
    def forward(self, x, attention_mask):
        x = x + self.multi_heads(self.ln1(x), attention_mask)
        out = x + self.ffw(self.ln2(x))
        return out
    

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_size)
        self.positional_embedding = nn.Embedding(config.block_size, config.embed_size)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_size)
        self.linear_f = nn.Linear(config.embed_size, config.vocab_size, bias=config.bias)
        self.linear_f.weight = self.token_embedding.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, attention_mask=None):
        B, T = idx.shape

        token_embed = self.token_embedding(idx)
        pos_embed = self.positional_embedding(torch.arange(T, device=device))

        x = token_embed + pos_embed
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.ln_f(x)
        logits = self.linear_f(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_generate_token, last_tokens):
        generated_tokens = []
        for i in range(max_generate_token):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(idx_next)
            idx = torch.cat((idx, idx_next), dim=-1)
            if idx[:, -3:].equal(last_tokens):
                break
        generated_tokens = torch.cat(generated_tokens, dim=-1)
        generated_tokens = generated_tokens[:, :-3]
        return generated_tokens



if __name__ == "__main__":
    gpt_config = GPTConfig()
    B, T, C = 3, 4, gpt_config.embed_size

    embed_x = torch.randn((B, T, gpt_config.embed_size))
    x = torch.randint(0, 9, (3, 4), dtype=torch.long)
    y = torch.randint(0, 9, (3, 4), dtype=torch.long)
    attention_mask = torch.ones((B, T))
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    gpt = GPT(gpt_config)


    logits, loss = gpt(x, y, attention_mask)
    num_params = sum(p.numel() for p in gpt.parameters())
    num_trainable_params = sum(p.numel() for p in gpt.parameters() if p.requires_grad)
    print(f"Num params: {num_params}")
    print(f"Num trainable params: {num_trainable_params}")


    print(f"GPT logits: {logits.shape}")
    print(f"GPT loss: {loss.item():.4f}")


    


