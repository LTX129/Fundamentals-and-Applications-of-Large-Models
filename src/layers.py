import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, max_len:int=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x):
        # x: (B, L, D)
        L = x.size(1)
        return x + self.pe[:, :L, :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dropout:float=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        B, Lq, D = q.size()
        _, Lk, _ = k.size()
        def split(x):
            return x.view(B, -1, self.n_heads, self.d_head).transpose(1,2)  # (B,H,L,Dh)
        q = split(self.w_q(q)); k = split(self.w_k(k)); v = split(self.w_v(v))
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_head)  # (B,H,Lq,Lk)
        bool_mask = None
        if attn_mask is not None:
            bool_mask = (attn_mask == 0)
        if key_padding_mask is not None:
            pad = (~key_padding_mask.bool()).unsqueeze(1).unsqueeze(2)  # (B,1,1,Lk), True=pad
            bool_mask = pad if bool_mask is None else (bool_mask | pad)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=bool_mask,  # True 表示屏蔽
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )
        out = attn_out.transpose(1, 2).contiguous().view(B, Lq, D)
        return self.w_o(out)

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model:int, ff_dim:int, dropout:float=0.1):
        super().__init__()
        self.lin1 = nn.Linear(d_model, ff_dim)
        self.lin2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.dropout(F.gelu(self.lin1(x))))

class PreNormResidual(nn.Module):
    def __init__(self, d_model:int, module:nn.Module, dropout:float=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.module = module
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, *args, **kwargs):
        h = self.module(self.norm(x), *args, **kwargs)
        return x + self.dropout(h)
