import torch
from loc4pm.models.bilstm_attn import BiLSTMAttnRegressor

def test_forward_smoke():
    B, T, F = 4, 5, 8
    model = BiLSTMAttnRegressor(input_size=F)
    x = torch.randn(B, T, F)
    y, attn = model(x)
    assert y.shape == (B,)
    assert attn.shape == (B, T)