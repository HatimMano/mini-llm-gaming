# src/models/components/rope.py
import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE)
    Encode la position par rotation dans l'espace complexe
    """
    def __init__(self, d_head: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_head = d_head
        
        # Calcul des fréquences de rotation
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer('inv_freq', inv_freq)
        
        # Pré-calcul des positions pour optimisation
        self._build_cos_sin_cache(max_seq_len)
    
    def _build_cos_sin_cache(self, seq_len: int):
        """Pré-calcule cos et sin pour les positions"""
        t = torch.arange(seq_len).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        
        # Duplication pour traiter les paires (real, imag)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotation de 90 degrés dans l'espace complexe"""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, 
                start_pos: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applique RoPE aux queries et keys
        Args:
            q, k: tenseurs [batch, n_heads, seq_len, d_head]
            start_pos: position de début (pour generation)
        """
        seq_len = q.shape[2]  # seq_len est à l'index 2
        
        # Récupération cos/sin pour les positions courantes
        cos = self.cos_cached[start_pos:start_pos + seq_len]  # [seq_len, d_head]
        sin = self.sin_cached[start_pos:start_pos + seq_len]  # [seq_len, d_head]
        
        # Ajout des dimensions pour broadcasting: [1, 1, seq_len, d_head]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Application de la rotation
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        
        return q_rot, k_rot