# config/ttt_config.py
from dataclasses import dataclass

@dataclass
class TTTNanoLLMConfig:
    """Configuration optimisée pour Tic-Tac-Toe NanoLLM"""
    # Architecture
    vocab_size: int = 16  # Réduit pour TTT
    d_model: int = 64     # Réduit car problème simple
    n_layers: int = 2     # Suffisant pour TTT
    n_heads: int = 2      
    d_ff: int = 256       # Réduit proportionnellement
    max_seq_len: int = 32 # Une partie complète + marge
    
    # Paramètres d'attention
    d_head: int = 32      # d_model // n_heads
    dropout: float = 0.0  
    
    # Quantification
    use_1bit_weights: bool = True
    activation_bits: int = 8
    
    # Optimisations
    use_rope: bool = True
    use_squared_relu: bool = True
    tie_embeddings: bool = True
    
    # Tokens spéciaux
    PAD_TOKEN = 0
    EMPTY_TOKEN = 1
    X_TOKEN = 2
    O_TOKEN = 3
    STATE_TOKEN = 4
    ACTION_TOKEN = 5
    POS_TOKENS = list(range(6, 15))  # Positions 0-8
    END_TOKEN = 15
