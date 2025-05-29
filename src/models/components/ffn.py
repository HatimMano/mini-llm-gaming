# src/models/components/ffn.py
import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    """
    Feed-Forward Network avec ReLU² pour compatibilité quantification 1-bit
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        
        # Deux couches linéaires sans biais
        self.linear1 = nn.Linear(self.d_model, self.d_ff, bias=False)
        self.linear2 = nn.Linear(self.d_ff, self.d_model, bias=False)
        
        # Activation
        self.use_squared_relu = config.use_squared_relu
    
    def squared_relu(self, x: torch.Tensor) -> torch.Tensor:
        """ReLU² activation: max(0, x)²"""
        return torch.square(torch.relu(x))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass du FFN"""
        x = self.linear1(x)
        
        if self.use_squared_relu:
            x = self.squared_relu(x)
        else:
            x = torch.relu(x)
        
        x = self.linear2(x)
        return x


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