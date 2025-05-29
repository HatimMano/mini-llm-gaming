import torch
import torch.nn as nn
from typing import Optional
from .components.attention import MultiHeadAttention
from .components.ffn import FeedForward
from .components.quantization import BitLinear 

class NanoLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        
        self.pos_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Transformer blocks avec support 1-bit
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_head=config.d_head,
                d_ff=config.d_ff,
                dropout=config.dropout,
                use_rope=config.use_rope,
                use_squared_relu=config.use_squared_relu,
                use_1bit=config.use_1bit_weights,  # NOUVEAU: Passer le flag 1-bit
            )
            for _ in range(config.n_layers)
        ])
        
        # Final norm - utilise LayerNorm standard
        self.ln_f = nn.LayerNorm(config.d_model, bias=False)
        
        # Output projection - peut être quantifié ou non
        if config.tie_embeddings:
            self.lm_head = lambda x: x @ self.token_embeddings.weight.T
        else:
            if config.use_1bit_weights:
                self.lm_head = BitLinear(config.d_model, config.vocab_size)
            else:
                self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    
    def get_quantization_stats(self) -> dict:
        """
        NOUVEAU: Méthode pour obtenir les stats de quantification globales.
        Utile pour monitoring et debugging.
        """
        if not self.config.use_1bit_weights:
            return {"error": "Model not using 1-bit quantization"}
        
        stats = {
            'layers': {},
            'global': {
                'total_params': 0,
                'quantized_params': 0,
                'embedding_params': 0,
                'average_sparsity': 0
            }
        }
        
        # Stats par couche
        total_sparsity = 0
        total_quantized = 0
        
        for i, block in enumerate(self.blocks):
            layer_stats = {}
            
            # Attention stats
            attn_stats = block.attention.get_bit_stats()
            layer_stats['attention'] = attn_stats
            
            # FFN stats
            ffn_stats = block.ffn.get_bit_stats()
            layer_stats['ffn'] = ffn_stats
            
            stats['layers'][f'layer_{i}'] = layer_stats
            
            # Accumulation pour stats globales
            if 'total' in ffn_stats:
                total_sparsity += ffn_stats['total']['sparsity']
                total_quantized += ffn_stats['total']['params']
            if 'qkv_proj' in attn_stats:
                total_sparsity += attn_stats['qkv_proj']['sparsity']
                total_quantized += attn_stats['qkv_proj']['total_params']
        
        # Stats globales
        embedding_params = self.token_embeddings.weight.numel()
        stats['global']['embedding_params'] = embedding_params
        stats['global']['quantized_params'] = total_quantized
        stats['global']['total_params'] = sum(p.numel() for p in self.parameters())
        stats['global']['average_sparsity'] = total_sparsity / len(self.blocks)
        stats['global']['compression_ratio'] = 32 / 1.58
        
        # Mémoire économisée
        original_size_mb = stats['global']['total_params'] * 4 / (1024 * 1024)  # float32
        quantized_size_mb = (
            embedding_params * 4 +  # embeddings restent float32
            total_quantized * 1.58 / 8
        ) / (1024 * 1024)
        stats['global']['original_size_mb'] = original_size_mb
        stats['global']['quantized_size_mb'] = quantized_size_mb
        stats['global']['size_reduction'] = 1 - quantized_size_mb / original_size_mb
        
        return stats


class TransformerBlock(nn.Module):
    """
    Bloc transformer modifié pour supporter 1-bit.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        d_ff: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        use_squared_relu: bool = True,
        use_1bit: bool = True,  # NOUVEAU: Support 1-bit
    ):
        super().__init__()
        
        # Attention avec potentielle quantification
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            dropout=dropout,
            use_rope=use_rope,
            use_1bit=use_1bit,  # NOUVEAU: Passer le flag
        )
        
        # FFN avec potentielle quantification
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            use_squared_relu=use_squared_relu,
            use_1bit=use_1bit,  # NOUVEAU: Passer le flag
        )
        
        # Normalisation - reste en pleine précision
        self.ln1 = nn.LayerNorm(d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model, bias=False)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention avec residual
        attn_out = self.attention(self.ln1(x), mask)
        x = x + attn_out
        
        # FFN avec residual  
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out
        
        return x