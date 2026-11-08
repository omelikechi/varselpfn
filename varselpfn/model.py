# Network
'''
Transformer mapping a dataset (X, y) to three logits per covariate: conditional dependence on y given the other
covariates, parent of y, and child of y. CD-PFN is trained on the first output and PC-PFN on the other two.

Every column (including y) is standardized within the dataset, and each cell is embedded by a shared scalar encoder
plus an embedding marking the response column. Each block applies attention across covariates within each row, then
across rows within each covariate, then a feedforward layer (pre-layer-norm residuals). Embeddings are pooled over
rows (mean and max, concatenated), and a shared MLP maps [h_j, h_y] to the three logits. There are no row-position or
covariate-index embeddings, so the output is invariant to row order and equivariant to covariate order.

Attention uses the parameters of nn.MultiheadAttention but runs through torch's fused scaled_dot_product_attention,
which does not store the full attention matrices: on CPU, nn.MultiheadAttention's own path does, which needs memory
quadratic in n (about 17 GB at n = 8000) and is several times slower. At inference (no gradients), each block runs in
slices (of rows for the attention across covariates and the feedforward layer, of covariates for the attention across
rows) and updates the embeddings in place, so the peak memory is about one copy of the (n, p, dim) embeddings
instead of several; the outputs are the same.

Architecture adapted from nanoTabPFN (https://github.com/automl/nanoTabPFN, Apache-2.0, commit 6071eec) and
AVICI (https://github.com/larslorch/avici, MIT, commit 40fb30d); code written from scratch.
'''

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

#----------------------------------------------------------------
# Layers
#----------------------------------------------------------------
def standardize_columns(Z, eps=1e-6):
    '''Z: (batch, rows, columns). Standardize each column within each dataset.'''
    mean = Z.mean(dim=1, keepdim=True)
    std = Z.std(dim=1, keepdim=True)
    return (Z - mean) / (std + eps)

def attention(mha, x, training):
    '''Self-attention of x (batch, length, dim) with the parameters of mha, through the fused kernel.'''
    b, length, d = x.shape
    h = mha.num_heads
    qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias).reshape(b, length, 3, h, d // h).permute(2, 0, 3, 1, 4)
    out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], dropout_p=mha.dropout if training else 0.0)
    return mha.out_proj(out.transpose(1, 2).reshape(b, length, d))

class Block(nn.Module):
    def __init__(self, dim, heads, ff_dim, dropout):
        super().__init__()
        self.norm_var = nn.LayerNorm(dim)
        self.attn_var = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_row = nn.LayerNorm(dim)
        self.attn_row = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        if not (self.training or torch.is_grad_enabled()):
            return self.forward_sliced(h)
        b, n, m, d = h.shape
        # attention across covariates within each row
        x = self.norm_var(h).reshape(b * n, m, d)
        a = attention(self.attn_var, x, self.training).reshape(b, n, m, d)
        h = h + self.dropout(a)
        # attention across rows within each covariate
        x = self.norm_row(h).transpose(1, 2).reshape(b * m, n, d)
        a = attention(self.attn_row, x, self.training).reshape(b, m, n, d).transpose(1, 2)
        h = h + self.dropout(a)
        # feedforward
        h = h + self.dropout(self.ff(self.norm_ff(h)))
        return h

    def forward_sliced(self, h, cells=65536):
        '''forward without gradients, in slices of about cells (row, covariate) pairs, updating h in place.'''
        b, n, m, d = h.shape
        rows = max(1, cells // m)
        for i in range(0, n, rows):
            s = h[:, i:i + rows]
            r = s.shape[1]
            s += attention(self.attn_var, self.norm_var(s).reshape(b * r, m, d), False).reshape(b, r, m, d)
        cols = max(1, cells // n)
        for j in range(0, m, cols):
            s = h[:, :, j:j + cols]
            c = s.shape[2]
            x = self.norm_row(s).transpose(1, 2).reshape(b * c, n, d)
            s += attention(self.attn_row, x, False).reshape(b, c, n, d).transpose(1, 2)
        for i in range(0, n, rows):
            s = h[:, i:i + rows]
            s += self.ff(self.norm_ff(s))
        return h

#----------------------------------------------------------------
# Model
#----------------------------------------------------------------
class Network(nn.Module):
    def __init__(self, dim=192, heads=6, ff_dim=768, blocks=8, dropout=0.0, pool='mean_max', grad_checkpoint=False):
        super().__init__()
        self.pool = pool
        # recompute block activations in the backward pass to save memory on large datasets
        self.grad_checkpoint = grad_checkpoint
        pooled_dim = 2 * dim if pool == 'mean_max' else dim
        self.value_encoder = nn.Linear(1, dim)
        self.role_embedding = nn.Embedding(2, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, ff_dim, dropout) for _ in range(blocks)])
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(2 * pooled_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, 3))

    def set_output_bias(self, prevalence):
        '''Initialize output logits at the prior label prevalence.'''
        prevalence = torch.as_tensor(prevalence, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.log(prevalence / (1 - prevalence)))

    def pooling(self, h):
        if self.pool == 'mean':
            return h.mean(dim=1)
        if self.pool == 'max':
            return h.amax(dim=1)
        return torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)

    def forward(self, X, y):
        '''X: (batch, n, p), y: (batch, n). Returns logits (batch, p, 3) for (conditional dependence, parent, child).'''
        Z = standardize_columns(torch.cat([X, y.unsqueeze(-1)], dim=-1))
        m = Z.shape[-1]
        role = torch.zeros(m, dtype=torch.long, device=Z.device)
        role[-1] = 1
        h = self.value_encoder(Z.unsqueeze(-1))
        h += self.role_embedding(role)
        for block in self.blocks:
            h = checkpoint(block, h, use_reentrant=False) if self.grad_checkpoint and self.training else block(h)
        # final norm and pooling over rows, in slices of covariates without gradients (same result)
        step = m if self.training or torch.is_grad_enabled() else max(1, 65536 // Z.shape[1])
        pooled = torch.cat([self.pooling(self.final_norm(h[:, :, j:j + step])) for j in range(0, m, step)], dim=1)
        h_x = pooled[:, :-1]
        h_y = pooled[:, -1:].expand_as(h_x)
        return self.head(torch.cat([h_x, h_y], dim=-1))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
