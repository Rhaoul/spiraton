import torch
import torch.nn as nn

class OperatorEmbeddings(nn.Module):
    """
    Embeddings par opérateur (Phase 3).
    Au lieu d'un vocabulaire plat, on utilise des tables d'embedding distinctes 
    pour chaque opérateur (ADD, SUB, MUL, DIV) et chiralité (Dextro, Levo).
    Les dimensions 0 à 5 du vecteur 33D (scores des opérateurs + orientation)
    pondèrent ces tables, conditionnant la représentation.
    """
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        
        # Tables pour Dextrogyre (DX)
        self.emb_add_dx = nn.Parameter(torch.randn(output_dim))
        self.emb_sub_dx = nn.Parameter(torch.randn(output_dim))
        self.emb_mul_dx = nn.Parameter(torch.randn(output_dim))
        self.emb_div_dx = nn.Parameter(torch.randn(output_dim))
        
        # Tables pour Lévogyre (LV)
        self.emb_add_lv = nn.Parameter(torch.randn(output_dim))
        self.emb_sub_lv = nn.Parameter(torch.randn(output_dim))
        self.emb_mul_lv = nn.Parameter(torch.randn(output_dim))
        self.emb_div_lv = nn.Parameter(torch.randn(output_dim))

    def forward(self, vector33d: torch.Tensor) -> torch.Tensor:
        """
        vector33d: tenseur de forme (..., 33)
        Retourne: (..., output_dim)
        """
        if vector33d.size(-1) != 33:
            raise ValueError(f"Attendu dim=33, reçu {vector33d.size(-1)}")
            
        scores_op = vector33d[..., 0:4]  # (ADD, SUB, MUL, DIV)
        score_dx = vector33d[..., 4].unsqueeze(-1)
        score_lv = vector33d[..., 5].unsqueeze(-1)
        
        w_add_dx = scores_op[..., 0:1] * score_dx
        w_sub_dx = scores_op[..., 1:2] * score_dx
        w_mul_dx = scores_op[..., 2:3] * score_dx
        w_div_dx = scores_op[..., 3:4] * score_dx
        
        w_add_lv = scores_op[..., 0:1] * score_lv
        w_sub_lv = scores_op[..., 1:2] * score_lv
        w_mul_lv = scores_op[..., 2:3] * score_lv
        w_div_lv = scores_op[..., 3:4] * score_lv
        
        out = (
            w_add_dx * self.emb_add_dx +
            w_sub_dx * self.emb_sub_dx +
            w_mul_dx * self.emb_mul_dx +
            w_div_dx * self.emb_div_dx +
            w_add_lv * self.emb_add_lv +
            w_sub_lv * self.emb_sub_lv +
            w_mul_lv * self.emb_mul_lv +
            w_div_lv * self.emb_div_lv
        )
        
        return out
