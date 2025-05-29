# src/models/components/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .rope import RotaryPositionalEmbedding

class MultiHeadAttention(nn.Module):
    """
    Attention multi-têtes optimisée pour modèles compacts
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.scale = 1.0 / math.sqrt(self.d_head)
        
        # Vérification de cohérence des dimensions
        assert self.d_model == self.n_heads * self.d_head, \
            f"d_model ({self.d_model}) doit être égal à n_heads * d_head ({self.n_heads * self.d_head})"
        
        # Projections Q, K, V en une seule matrice pour efficacité
        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        # RoPE si activé
        if config.use_rope:
            self.rope = RotaryPositionalEmbedding(self.d_head, config.max_seq_len)
        else:
            self.rope = None
        
        # Cache du masque causal pour éviter les recalculs
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        )
    
    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Forward pass de l'attention
        Args:
            x: [batch, seq_len, d_model]
            start_pos: position pour génération incrémentale
        """
        batch_size, seq_len, _ = x.shape
        
        # Projection Q, K, V
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3 * d_model]
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape pour multi-head: [batch, seq_len, n_heads, d_head]
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head)
        
        # Transposition pour attention: [batch, n_heads, seq_len, d_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Application RoPE si activée (APRÈS la transposition)
        if self.rope is not None:
            q, k = self.rope(q, k, start_pos)
        
        # Calcul des scores d'attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Application du masque causal
        if seq_len > 1:
            mask = self.causal_mask[:seq_len, :seq_len]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax et application aux values
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        # Reshape et projection finale
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        
        return out