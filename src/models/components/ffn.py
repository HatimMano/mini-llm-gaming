# src/models/ffn.py
import torch
import torch.nn as nn
from .quantization import BitLinear  # NOUVEAU: Import BitLinear

class FeedForward(nn.Module):
    """
    Feed-forward network avec support optionnel pour quantification 1-bit.
    
    Changements pour 1-bit:
    - Les deux couches Linear peuvent être remplacées par BitLinear
    - Économie massive de mémoire: d_model * d_ff * 2 paramètres réduits à 1.58 bits
    - Performance: multiplications remplacées par additions ternaires
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        use_squared_relu: bool = True,
        use_1bit: bool = True,  # NOUVEAU: Paramètre pour activer 1-bit
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_squared_relu = use_squared_relu
        self.use_1bit = use_1bit  # NOUVEAU: Stocker le choix
        
        # MODIFICATION: Choix entre Linear standard et BitLinear
        if self.use_1bit:
            # BitLinear pour l'expansion d_model -> d_ff
            # C'est ici que la majorité des paramètres sont économisés
            self.linear1 = BitLinear(self.d_model, self.d_ff)
            
            # BitLinear pour la projection d_ff -> d_model
            self.linear2 = BitLinear(self.d_ff, self.d_model)
        else:
            # Garder l'implémentation standard pour comparaison
            self.linear1 = nn.Linear(self.d_model, self.d_ff, bias=False)
            self.linear2 = nn.Linear(self.d_ff, self.d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass inchangé - BitLinear gère la quantification en interne.
        
        Note importante pour 1-bit:
        - L'activation (ReLU²) reste en pleine précision
        - Seuls les poids sont quantifiés
        - Cela préserve la capacité d'expression du réseau
        """
        # Expansion avec potentielle quantification
        x = self.linear1(x)
        
        # Activation - reste en pleine précision
        if self.use_squared_relu:
            x = torch.square(torch.relu(x))
        else:
            x = torch.relu(x)
        
        x = self.dropout(x)
        
        # Projection avec potentielle quantification
        x = self.linear2(x)
        
        return x
    
    def get_bit_stats(self) -> dict:
        """
        NOUVEAU: Méthode pour obtenir les statistiques de quantification.
        
        Particulièrement intéressant pour FFN car:
        - linear1 a beaucoup de paramètres (d_model * d_ff)
        - On peut observer la sparsité qui émerge naturellement
        """
        if not self.use_1bit:
            return {"error": "Not using 1-bit quantization"}
        
        stats = {}
        
        with torch.no_grad():
            # Stats pour linear1 (expansion)
            w1 = self.linear1.weight
            q1 = self.linear1.quantize_weights(w1)
            
            stats['linear1'] = {
                'shape': list(w1.shape),
                'total_params': w1.numel(),
                'zeros': (q1 == 0).sum().item(),
                'positive': (q1 == 1).sum().item(),
                'negative': (q1 == -1).sum().item(),
                'sparsity': (q1 == 0).float().mean().item(),
                'compression_ratio': 32 / 1.58  # bits saved
            }
            
            # Stats pour linear2 (projection)
            w2 = self.linear2.weight
            q2 = self.linear2.quantize_weights(w2)
            
            stats['linear2'] = {
                'shape': list(w2.shape),
                'total_params': w2.numel(),
                'zeros': (q2 == 0).sum().item(),
                'positive': (q2 == 1).sum().item(),
                'negative': (q2 == -1).sum().item(),
                'sparsity': (q2 == 0).float().mean().item(),
                'compression_ratio': 32 / 1.58
            }
            
            # Stats globales FFN
            total_params = w1.numel() + w2.numel()
            total_zeros = (q1 == 0).sum().item() + (q2 == 0).sum().item()
            
            stats['total'] = {
                'params': total_params,
                'sparsity': total_zeros / total_params,
                'memory_saved_mb': (total_params * 32 - total_params * 1.58) / (8 * 1024 * 1024)
            }
        
        return stats