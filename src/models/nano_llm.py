# src/models/nano_llm.py
import torch
import torch.nn as nn
from .base_model import BaseLLM
from .components.embeddings import TokenEmbedding
from .components.attention import MultiHeadAttention
from .components.ffn import FeedForwardNetwork

class TransformerBlock(nn.Module):
    """Bloc transformer pour NanoLLM"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, bias=False)
        self.attention = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model, bias=False)
        self.ffn = FeedForwardNetwork(config)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Attention avec connexion résiduelle
        attn_out = self.attention(self.ln1(x), **kwargs)
        x = x + attn_out
        
        # FFN avec connexion résiduelle
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out
        
        return x

class NanoLLM(BaseLLM):
    """
    NanoLLM: Architecture ultra-légère
    - 2 couches transformer
    - 2 têtes d'attention
    - 128 dimensions
    - ~1M paramètres, ~0.3MB quantifié
    """
    def __init__(self, config):
        super().__init__(config)
        
        # Embedding des tokens
        self.token_embedding = TokenEmbedding(config)
        
        # Blocs transformer
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Normalisation finale
        self.ln_final = nn.LayerNorm(config.d_model, bias=False)
        
        # Tête de sortie
        if config.tie_embeddings:
            # Partage des poids avec l'embedding
            self.output_projection = None
        else:
            self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialisation
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialisation des poids optimisée pour petits modèles"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass
        Args:
            input_ids: [batch, seq_len]
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Embedding
        x = self.token_embedding(input_ids)
        
        # Passage dans les blocs transformer
        for block in self.transformer_blocks:
            x = block(x, **kwargs)
        
        # Normalisation finale
        x = self.ln_final(x)
        
        # Projection vers vocabulaire
        if self.output_projection is not None:
            logits = self.output_projection(x)
        else:
            # Utilisation des poids d'embedding partagés
            logits = torch.matmul(x, self.token_embedding.embedding.weight.T)
        
        return logits