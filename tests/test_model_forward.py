import torch
from loc4pm.models.bilstm_attn import BiLSTMAttnRegressor

def test_forward_smoke():
    B, T, F = 4, 5, 8
    model = BiLSTMAttnRegressor(input_size=F)
    x = torch.randn(B, T, F)
    y, attn = model(x)
    assert y.shape == (B,)
    assert attn.shape == (B, T)

def test_forward_loc_fusion_smoke():
    """Sanity check for the location-aware BiLSTM model.

    This test ensures that the extended model can accept time-series
    inputs along with coordinate tensors (and optionally month indices)
    and produce outputs of the expected shape. We deliberately choose
    a configuration that falls back to the internal MLP-based encoder
    when external dependencies are unavailable.
    """
    from loc4pm.models.bilstm_attn_fusion import BiLSTMAttnLocRegressor

    B, T, F = 3, 7, 6
    coords = torch.randn(B, 2)
    month = torch.randint(1, 13, (B,))
    # Instantiate model with location encoder enabled but using a fallback MLP.
    model = BiLSTMAttnLocRegressor(
        input_size=F,
        hidden_size=4,
        num_layers=1,
        bidirectional=False,
        dropout=0.0,
        layer_norm=True,
        attn_type='luong',
        attn_dim=4,
        loc_name='climplicit',  # will fall back to MLP if rshf is unavailable
        loc_variant='monthly',
        loc_emb_dim=4,
        loc_pretrained=False,
        loc_freeze=False,
        loc_proj_dim=4,
        fusion_method='concat',
        fusion_hidden_dim=4,
    )
    x = torch.randn(B, T, F)
    y, attn = model(x, coords, month)
    assert y.shape == (B,)
    assert attn.shape == (B, T)