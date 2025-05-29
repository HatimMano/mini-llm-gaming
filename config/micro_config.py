# config/micro_config.py
from dataclasses import dataclass

@dataclass
class MicroLLMConfig:
    """Configuration pour MicroLLM - Architecture légère mais plus capable"""
    # Architecture
    vocab_size: int = 8000
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    max_seq_len: int = 512
    
    # Paramètres d'attention
    d_head: int = 64  # d_model // n_heads
    dropout: float = 0.0
    
    # Quantification
    use_1bit_weights: bool = True
    activation_bits: int = 8
    
    # Optimisations
    use_rope: bool = True
    use_squared_relu: bool = True
    tie_embeddings: bool = True