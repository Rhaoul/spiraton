import torch
import torch.nn as nn
from spiraton.core.cell import MatrixSpiratonCell
from spiraton.core.embeddings import OperatorEmbeddings
from spiraton.core.mode_policy import LearnedGateMode
from spiraton.recursion.recursive import RecursiveSpiraton
from spiraton.experimental.chrono import ChronoSpiraton

class ABAModel(nn.Module):
    """
    Modèle d'entraînement pour la boucle A -> B -> A'.
    Incorpore les OperatorEmbeddings et la cellule MatrixSpiratonCell non-commutative.
    """
    def __init__(self, dim: int = 64, state_size: int = 64, use_chrono: bool = False, pure_phonetic: bool = False):
        super().__init__()
        self.use_chrono = use_chrono
        self.pure_phonetic = pure_phonetic
        self.state_size = state_size
        
        x_size = 15 if pure_phonetic else dim
        
        if not pure_phonetic:
            self.embeddings = OperatorEmbeddings(dim)
        
        if use_chrono:
            self.chrono = ChronoSpiraton(x_size=x_size, state_size=state_size, dt_init=1.0)
        else:
            # Phase 5 : Stabilisation Gated via LearnedGateMode (gestion fine des bascules)
            mode_policy = LearnedGateMode(input_size=x_size + state_size)
            cell = MatrixSpiratonCell(input_size=x_size + state_size, mode_policy=mode_policy)
            self.recursion = RecursiveSpiraton(
                cell=cell,
                x_size=x_size,
                state_size=state_size,
                combine="concat",
                update="residual",
                cell_out_size=x_size + state_size
            )
        self.state_to_pred = nn.Linear(state_size, x_size)
        
    def forward(self, a_vecs: torch.Tensor, b_vecs: torch.Tensor):
        """
        a_vecs: (B, seq_len, 33)
        b_vecs: (B, seq_len, 33)
        """
        if self.pure_phonetic:
            # Extraction directe des 15 dimensions de flux, impédance et spins [8:23]
            a_emb = a_vecs[..., 8:23] # (B, seq_len, 15)
            b_emb = b_vecs[..., 8:23]
        else:
            # 1. Embeddings (pondérés par les opérateurs)
            a_emb = self.embeddings(a_vecs) # (B, seq_len, dim)
            b_emb = self.embeddings(b_vecs)
        
        # Agrégation (moyenne)
        a_agg = a_emb.mean(dim=1) # (B, dim)
        b_agg = b_emb.mean(dim=1) # (B, dim)
        
        # 2. Boucle Récursive (A -> B)
        B_size = a_agg.size(0)
        device, dtype = a_agg.device, a_agg.dtype
        
        if self.use_chrono:
            s_t = torch.zeros(B_size, self.state_size, device=device, dtype=dtype)
            s_tm1 = torch.zeros(B_size, self.state_size, device=device, dtype=dtype)
            
            # Application de A
            s_next, s_t = self.chrono(a_agg, s_t, s_tm1)
            s_tm1 = s_t
            s_t = s_next
            
            # Application de B
            s_next, s_t = self.chrono(b_agg, s_t, s_tm1)
            state = s_next
        else:
            state = self.recursion.init_state(B_size, device=device, dtype=dtype)
            _, state, _ = self.recursion(a_agg, state=state, steps=1, return_trace=True)
            _, state, _ = self.recursion(b_agg, state=state, steps=1, return_trace=True)
        
        # 3. Génération de A' (Retour)
        a_prime_pred = self.state_to_pred(state)
        
        return a_prime_pred, a_agg
