# src/models/components/embeddings.py
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """
    Embedding des tokens avec support du partage de poids
    """
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Initialisation adaptée aux petits modèles
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq_len]
        Returns:
            embeddings: [batch, seq_len, d_model]
        """
        return self.embedding(token_ids)