import os
import torch
import torch.nn as nn
from torch.nn import functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Head(nn.Module):
    def __init__(self, embed_size, block_size, head_size, dropout=0.3):
        super().__init__()
        self.query = nn.Linear(embed_size, head_size)
        self.key = nn.Linear(embed_size, head_size)
        self.value = nn.Linear(embed_size, head_size)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)

        causal_mask = self.tril[:T, :T].unsqueeze(0).expand(B, -1, -1)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 
        wei = wei.masked_fill(causal_mask==0, float('-inf'))

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).expand(B, T, T)
            wei = wei.masked_fill(attention_mask==0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        

        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, head_size, block_size, num_heads, dropout=0.3):
        super().__init__()
        self.heads = nn.ModuleList([Head(embed_size, block_size, head_size, dropout=dropout) for _ in range(num_heads)])
        self.linear = nn.Linear(head_size*num_heads, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        x = torch.cat([head(x, attention_mask) for head in self.heads], dim=-1)
        x = self.linear(x)
        out = self.dropout(x)
        return out
    

class FeedForward(nn.Module):
    def __init__(self, embed_size, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, embed_size*4),
            nn.ReLU(),
            nn.Linear(embed_size*4, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class Block(nn.Module):
    def __init__(self, embed_size, block_size, num_heads, dropout=0.3):
        super().__init__()
        head_size = embed_size//num_heads
        self.multi_heads = MultiHeadAttention(embed_size, head_size, block_size, num_heads, dropout=dropout)
        self.ffw = FeedForward(embed_size, dropout=dropout)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        
    def forward(self, x, attention_mask):
        x = x + self.multi_heads(self.ln1(x), attention_mask)
        out = x + self.ffw(self.ln2(x))
        return out
    

class GPT(nn.Module):
    def __init__(self, embed_size, vocab_size, block_size, num_heads, num_layers, dropout=0.3):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(*[Block(embed_size, block_size, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_size)
        self.linear_f = nn.Linear(embed_size, vocab_size)
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
    block_size = 128
    embed_size = 64
    num_heads = 2
    num_layers = 2
    dropout = 0.3

    vocab_size = 888888 
    B, T, C = 3, 4, embed_size

    embed_x = torch.randn((B, T, embed_size))
    x = torch.randint(0, 9, (3, 4), dtype=torch.long)
    y = torch.randint(0, 9, (3, 4), dtype=torch.long)
    attention_mask = torch.ones((B, T))
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    gpt = GPT(embed_size, vocab_size, block_size, num_heads, num_layers, dropout=dropout)


    logits, loss = gpt(x, y, attention_mask)
    g = gpt.generate(context, 10)


    print("GPT logits: ", logits.shape)
    print("GPT loss: ", loss)
    print("GPT generate: ", g.shape)

    


