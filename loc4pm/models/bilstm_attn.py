import torch
import torch.nn as nn
from typing import Optional

class LuongAttention(nn.Module):
    def __init__(self, hidden_dim, attn_dim=None):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim, bias=False) if attn_dim else None
        self.score = nn.Linear(attn_dim, 1, bias=False) if attn_dim else None

    def forward(self, outputs, query, mask=None):
        # outputs: [B,T,H], query: [B,H]
        if self.proj is not None:
            e = self.score(torch.tanh(self.proj(outputs))).squeeze(-1)  # [B,T]
        else:  # dot(q, outputs_t)
            e = torch.bmm(outputs, query.unsqueeze(2)).squeeze(-1)
        if mask is not None:
            e = e.masked_fill(mask == 0, float('-inf'))
        a = torch.softmax(e, dim=-1)         # [B,T]
        ctx = torch.bmm(a.unsqueeze(1), outputs).squeeze(1)  # [B,H]
        return ctx, a

class BiLSTMAttnRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, bidirectional=True,
                 dropout=0.2, layer_norm=True, attn_type='luong', attn_dim=256,
                 head_dropout: Optional[float] = None):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers>1 else 0.0,
                            bidirectional=bidirectional)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()
        self.attn = LuongAttention(out_dim, attn_dim if attn_type=='luong' else None)
        # Choose a separate dropout rate for the prediction head if provided.  If not, reuse the
        # sequence dropout to preserve backward compatibility.  A modest dropout (e.g. 0.1–0.2)
        # can improve generalization for the final regression layer.
        hd = dropout if head_dropout is None else float(head_dropout)
        self.head = nn.Sequential(nn.Dropout(hd), nn.Linear(out_dim, 1))

    def forward(self, x, mask=None):
        out, (h, _) = self.lstm(x)       # out: [B,T,H*dir]
        out = self.norm(out)
        q = torch.cat([h[-2], h[-1]], dim=-1) if self.lstm.bidirectional else h[-1]
        ctx, attn = self.attn(out, q, mask)
        y = self.head(ctx).squeeze(-1)
        return y, attn