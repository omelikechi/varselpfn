# Network
'''
Transformer mapping a dataset (X, y) to three logits per covariate: conditional dependence on y given the other
covariates, parent of y, and child of y. CD-PFN is trained on the first output and PC-PFN on the other two.

Every column (including y) is standardized within the dataset, and each cell is embedded by a shared scalar encoder
plus an embedding marking the response column. Each block applies attention across covariates within each row, then
across rows within each covariate, then a feedforward layer (pre-layer-norm residuals). Embeddings are pooled over
rows (mean and max, concatenated), and a shared MLP maps [h_j, h_y] to the three logits. There are no row-position or
covariate-index embeddings, so the output is invariant to row order and equivariant to covariate order.

Architecture adapted from nanoTabPFN (https://github.com/automl/nanoTabPFN, Apache-2.0, commit 6071eec) and
AVICI (https://github.com/larslorch/avici, MIT, commit 40fb30d); code written from scratch.
'''

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

#----------------------------------------------------------------
# Layers
#----------------------------------------------------------------
def standardize_columns(Z, eps=1e-6):
    '''Z: (batch, rows, columns). Standardize each column within each dataset.'''
    mean = Z.mean(dim=1, keepdim=True)
    std = Z.std(dim=1, keepdim=True)
    return (Z - mean) / (std + eps)

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
        b, n, m, d = h.shape
        # attention across covariates within each row
        x = self.norm_var(h).reshape(b * n, m, d)
        a = self.attn_var(x, x, x, need_weights=False)[0].reshape(b, n, m, d)
        h = h + self.dropout(a)
        # attention across rows within each covariate
        x = self.norm_row(h).transpose(1, 2).reshape(b * m, n, d)
        a = self.attn_row(x, x, x, need_weights=False)[0].reshape(b, m, n, d).transpose(1, 2)
        h = h + self.dropout(a)
        # feedforward
        h = h + self.dropout(self.ff(self.norm_ff(h)))
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

    def forward(self, X, y):
        '''X: (batch, n, p), y: (batch, n). Returns logits (batch, p, 3) for (conditional dependence, parent, child).'''
        Z = standardize_columns(torch.cat([X, y.unsqueeze(-1)], dim=-1))
        m = Z.shape[-1]
        role = torch.zeros(m, dtype=torch.long, device=Z.device)
        role[-1] = 1
        h = self.value_encoder(Z.unsqueeze(-1)) + self.role_embedding(role)
        for block in self.blocks:
            h = checkpoint(block, h, use_reentrant=False) if self.grad_checkpoint and self.training else block(h)
        h = self.final_norm(h)
        if self.pool == 'mean':
            pooled = h.mean(dim=1)
        elif self.pool == 'max':
            pooled = h.amax(dim=1)
        else:
            pooled = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)
        h_x = pooled[:, :-1]
        h_y = pooled[:, -1:].expand_as(h_x)
        return self.head(torch.cat([h_x, h_y], dim=-1))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
