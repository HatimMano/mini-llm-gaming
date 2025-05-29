# src/models/attention.py
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .rope import RotaryPositionalEmbedding
from .quantization import BitLinear  # NOUVEAU: Import BitLinear

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention avec support optionnel pour quantification 1-bit.
    
    Changements pour 1-bit:
    - Remplacement des nn.Linear par BitLinear quand use_1bit=True
    - Les BitLinear remplacent les multiplications par des additions/soustractions
    - Réduction de la taille mémoire de 32 bits à 1.58 bits par poids
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        use_1bit: bool = True,  # NOUVEAU: Paramètre pour activer 1-bit
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.use_rope = use_rope
        self.use_1bit = use_1bit  # NOUVEAU: Stocker le choix
        
        # MODIFICATION: Choix entre Linear standard et BitLinear
        if self.use_1bit:
            # BitLinear pour projection Q, K, V combinée
            # Note: BitLinear n'a pas de paramètre bias (toujours False)
            self.qkv_proj = BitLinear(self.d_model, 3 * self.d_model)
            
            # BitLinear pour projection de sortie
            self.out_proj = BitLinear(self.d_model, self.d_model)
        else:
            # Garder l'implémentation standard pour comparaison
            self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
            self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5
        
        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(self.d_head, max_seq_len=2048)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass inchangé - BitLinear est transparent à l'utilisation.
        
        La quantification se fait automatiquement dans BitLinear:
        1. Forward: poids sont quantifiés à {-1, 0, 1}
        2. Backward: gradients passent normalement (Straight-Through Estimator)
        """
        batch_size, seq_len, _ = x.shape
        
        # Projection QKV - maintenant potentiellement quantifiée
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Rotary embeddings si activé
        if self.use_rope:
            q, k = self.rope(q, k, start_pos=0)
        
        # Attention standard
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Multiplication avec values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Projection de sortie - maintenant potentiellement quantifiée
        output = self.out_proj(attn_output)
        
        return output
    
    def get_bit_stats(self) -> dict:
        """
        NOUVEAU: Méthode pour obtenir les statistiques de quantification.
        Utile pour le monitoring pendant l'entraînement.
        """
        if not self.use_1bit:
            return {"error": "Not using 1-bit quantization"}
        
        stats = {}
        
        # Stats pour QKV projection
        with torch.no_grad():
            qkv_weights = self.qkv_proj.weight
            qkv_quantized = self.qkv_proj.quantize_weights(qkv_weights)
            
            stats['qkv_proj'] = {
                'total_params': qkv_weights.numel(),
                'zeros': (qkv_quantized == 0).sum().item(),
                'positive': (qkv_quantized == 1).sum().item(),
                'negative': (qkv_quantized == -1).sum().item(),
                'sparsity': (qkv_quantized == 0).float().mean().item()
            }
            
            # Stats pour output projection
            out_weights = self.out_proj.weight
            out_quantized = self.out_proj.quantize_weights(out_weights)
            
            stats['out_proj'] = {
                'total_params': out_weights.numel(),
                'zeros': (out_quantized == 0).sum().item(),
                'positive': (out_quantized == 1).sum().item(),
                'negative': (out_quantized == -1).sum().item(),
                'sparsity': (out_quantized == 0).float().mean().item()
            }
        
        return stats