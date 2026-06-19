import torch
import torch.nn as nn
from typing import Tuple, Optional

class ChronoSpiraton(nn.Module):
    """
    ChronoSpiraton (Expérimental - Phase 5)
    Incarne l'équation du second ordre du Logos Opératoire :
    S_{t+1} = (1 - dt)*S_t + dt * [ D( X(x_t) + A(S_t) + B(S_t^2) - C(S_{t-1}) ) + L(S_t) ]
    
    Cette cellule sépare formellement la branche Dextrogyre (Action/Expansion) 
    de la branche Lévogyre (Mémoire/Réception).
    """
    def __init__(
        self, 
        x_size: int, 
        state_size: int, 
        init_scale: float = 0.1,
        dt_init: float = 1.0,
        eps: float = 1e-6
    ):
        super().__init__()
        self.state_size = state_size
        self.eps = eps
        
        # Le pas de temps dt devient un paramètre apprenable pour le temps continu
        self.dt_raw = nn.Parameter(torch.tensor(float(dt_init)))
        
        # Projections linéaires (Espace d'état)
        self.proj_X = nn.Linear(x_size, state_size, bias=False)
        self.proj_A = nn.Linear(state_size, state_size, bias=False)
        self.proj_B = nn.Linear(state_size, state_size, bias=False)
        self.proj_C = nn.Linear(state_size, state_size, bias=False)
        
        # Poids matriciels pour l'opérateur Dextrogyre (D)
        self.w_add_D = nn.Parameter(torch.randn(state_size, state_size) * init_scale)
        self.w_mul_D = nn.Parameter(torch.randn(state_size, state_size) * (init_scale / 2.0))
        self.w_div_D = nn.Parameter(torch.randn(state_size, state_size) * (init_scale / 2.0))
        self.bias_D = nn.Parameter(torch.zeros(state_size))
        
        # Poids matriciels pour l'opérateur Lévogyre (L)
        self.w_sub_L = nn.Parameter(torch.randn(state_size, state_size) * init_scale)
        self.w_div_L = nn.Parameter(torch.randn(state_size, state_size) * (init_scale / 2.0))
        self.w_mul_L = nn.Parameter(torch.randn(state_size, state_size) * (init_scale / 2.0))
        self.bias_L = nn.Parameter(torch.zeros(state_size))

    def _dextro(self, u: torch.Tensor) -> torch.Tensor:
        """ Opérateur D : Expansion/Action (Tanh) """
        add = torch.matmul(u, self.w_add_D)
        
        log_abs = torch.log(torch.clamp(u.abs(), min=self.eps))
        mul = torch.tanh(torch.matmul(log_abs, self.w_mul_D))
        div = torch.tanh(-torch.matmul(log_abs, self.w_div_D))
        
        return torch.tanh(add + mul - div + self.bias_D)

    def _levo(self, s: torch.Tensor) -> torch.Tensor:
        """ Opérateur L : Mémoire/Contraction (Arctan) """
        sub = torch.matmul(s, self.w_sub_L)
        
        log_abs = torch.log(torch.clamp(s.abs(), min=self.eps))
        div = torch.tanh(-torch.matmul(log_abs, self.w_div_L))
        mul = torch.tanh(torch.matmul(log_abs, self.w_mul_L))
        
        return torch.atan(sub + div - mul + self.bias_L)

    def compute_components(self, x_t: torch.Tensor, s_t: torch.Tensor, s_tm1: torch.Tensor):
        u_x = self.proj_X(x_t)
        u_a = self.proj_A(s_t)
        u_b = self.proj_B(s_t * s_t)  # Terme quadratique B(s_t^2)
        u_c = self.proj_C(s_tm1)      # Terme mémoriel C(s_{t-1})
        
        u_t = u_x + u_a + u_b - u_c
        
        d_out = self._dextro(u_t)
        l_out = self._levo(s_t)
        
        return d_out, l_out, u_x, u_a, u_b, u_c

    def forward(
        self, 
        x_t: torch.Tensor, 
        s_t: Optional[torch.Tensor] = None, 
        s_tm1: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_t: (B, x_size)
        s_t: (B, state_size) ou None (initialisé à 0)
        s_tm1: (B, state_size) ou None (initialisé à 0)
        
        Retourne (s_t_plus_1, s_t)
        """
        B = x_t.size(0)
        device, dtype = x_t.device, x_t.dtype
        
        if s_t is None:
            s_t = torch.zeros(B, self.state_size, device=device, dtype=dtype)
        if s_tm1 is None:
            s_tm1 = torch.zeros(B, self.state_size, device=device, dtype=dtype)
            
        d_out, l_out, _, _, _, _ = self.compute_components(x_t, s_t, s_tm1)
        
        # 3. Intégration temporelle apprenable
        dt = torch.clamp(self.dt_raw, 0.0, 1.0)
        s_t_plus_1 = (1.0 - dt) * s_t + dt * (d_out + l_out)
        
        return s_t_plus_1, s_t
