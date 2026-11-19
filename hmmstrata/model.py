"""Small pre-LN GPT, Shai et al. style: n_layers x (1 head of d_head, ReLU MLP)."""
import torch, torch.nn as nn, torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, n_heads, d_head, d_mlp):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.nh, self.dh = n_heads, d_head
        self.qkv = nn.Linear(d, 3 * n_heads * d_head)
        self.o = nn.Linear(n_heads * d_head, d)
        self.fc1, self.fc2 = nn.Linear(d, d_mlp), nn.Linear(d_mlp, d)

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.nh, self.dh).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, -1))
        return x + self.fc2(F.relu(self.fc1(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, vocab, ctx=16, d=64, n_layers=4, n_heads=1, d_head=8, d_mlp=256):
        super().__init__()
        self.cfg = dict(vocab=vocab, ctx=ctx, d=d, n_layers=n_layers, n_heads=n_heads,
                        d_head=d_head, d_mlp=d_mlp)
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, n_heads, d_head, d_mlp) for _ in range(n_layers))
        self.lnf, self.head = nn.LayerNorm(d), nn.Linear(d, vocab)

    def forward(self, idx, return_resid=False):
        """return_resid: False | True (final residual) | 'all' (list: embedding + after each block)."""
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        allr = [x]
        for b in self.blocks:
            x = b(x)
            allr.append(x)
        logits = self.head(self.lnf(x))
        if return_resid == "all":
            return logits, allr
        return (logits, x) if return_resid else logits
