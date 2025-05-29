# config/nano_config.py
from dataclasses import dataclass

@dataclass
class NanoLLMConfig:
    """Configuration pour NanoLLM - Architecture ultra-légère"""
    # Architecture
    vocab_size: int = 5000
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 2
    d_ff: int = 512
    max_seq_len: int = 256
    
    # Paramètres d'attention
    d_head: int = 64  # d_model // n_heads
    dropout: float = 0.0  # Pas de dropout pour l'inférence
    
    # Quantification
    use_1bit_weights: bool = True
    activation_bits: int = 8
    
    # Optimisations
    use_rope: bool = True
    use_squared_relu: bool = True
    tie_embeddings: bool = True  # Partage poids embedding/output