import torch
import pytest
from spiraton.experimental.chrono import ChronoSpiraton

def test_chrono_spiraton_init():
    cell = ChronoSpiraton(x_size=32, state_size=64, dt_init=0.5)
    assert cell.state_size == 64
    assert cell.dt_raw.item() == 0.5

def test_chrono_spiraton_forward():
    cell = ChronoSpiraton(x_size=32, state_size=64)
    B = 4
    x_t = torch.randn(B, 32)
    s_t = torch.randn(B, 64)
    s_tm1 = torch.randn(B, 64)
    
    s_t_plus_1, s_t_out = cell(x_t, s_t, s_tm1)
    
    assert s_t_out is s_t
    assert s_t_plus_1.shape == (B, 64)
    assert not torch.isnan(s_t_plus_1).any()

def test_chrono_spiraton_zero_state():
    # If state is zero, L(0) might trigger log(0). 
    # With eps=1e-6, it should be stable.
    cell = ChronoSpiraton(x_size=32, state_size=16)
    x_t = torch.randn(2, 32)
    
    # Defaults to zero
    s_t_plus_1, s_t_out = cell(x_t)
    
    assert s_t_out.shape == (2, 16)
    assert (s_t_out == 0).all()
    assert s_t_plus_1.shape == (2, 16)
    assert not torch.isnan(s_t_plus_1).any()

def test_chrono_spiraton_unroll():
    cell = ChronoSpiraton(x_size=16, state_size=8, dt_init=0.1)
    
    B = 3
    s_t, s_tm1 = None, None
    seq_len = 10
    
    for t in range(seq_len):
        x_t = torch.randn(B, 16)
        s_next, s_t_out = cell(x_t, s_t, s_tm1)
        s_tm1 = s_t_out
        s_t = s_next
        
        assert not torch.isnan(s_t).any(), f"NaN at step {t}"
