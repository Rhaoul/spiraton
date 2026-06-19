import torch
import torch.nn as nn
from spiraton.diagnostics.alpha_omega_spatial import alpha_omega_metrics

class ABALoopLoss(nn.Module):
    """
    Objectif d'Apprentissage Alpha-Oméga (Phase 4).
    A' prédit doit être proche-et-aligné avec A sans lui être identique.
    La répétition (copie exacte) est pénalisée.
    """
    def __init__(self, eps: float = 1e-12, copy_penalty_weight: float = 0.1, supervised_weight: float = 1.0):
        super().__init__()
        self.eps = eps
        self.copy_penalty_weight = copy_penalty_weight
        self.supervised_weight = supervised_weight
        self.mse = nn.MSELoss()
        
    def forward(self, a_pred: torch.Tensor, a_true: torch.Tensor, a_prime_true: torch.Tensor) -> torch.Tensor:
        """
        a_pred: (B, D) A' prédit par le réseau
        a_true: (B, D) Le segment A originel (pour vérifier la clôture)
        a_prime_true: (B, D) Le segment A' cible (pour supervision)
        """
        # 1. Supervision classique sur A' cible
        sup_loss = self.mse(a_pred, a_prime_true)
        
        # 2. Clôture Alpha-Oméga: Comparaison entre A_pred et A originel
        l2, cos = alpha_omega_metrics(a_true, a_pred, eps=self.eps)
        
        # Aligné: cosinus proche de 1 -> on minimise 1 - cos
        alignment_loss = (1.0 - cos).mean()
        
        # Distance: on minimise L2 globalement
        distance_loss = l2.mean()
        
        # Mais pénalité très forte si l2 s'approche de 0 (copie exacte interdite)
        copy_penalty = (1.0 / (l2 + 0.01)).mean()
        
        loss = (self.supervised_weight * sup_loss) + alignment_loss + distance_loss + (self.copy_penalty_weight * copy_penalty)
        return loss
