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
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass du modèle NanoLLM
        
        Args:
            input_ids: Tensor de shape (batch_size, seq_len)
            attention_mask: Masque d'attention optionnel (batch_size, seq_len)
            
        Returns:
            logits: Tensor de shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        token_emb = self.token_embeddings(input_ids)  # (batch_size, seq_len, d_model)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embeddings(positions)  # (batch_size, seq_len, d_model)
        
        # Combiner embeddings
        x = token_emb + pos_emb
        
        # Passer par les blocs transformer
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Normalisation finale
        x = self.ln_f(x)
        
        # Projection vers vocabulaire
        if callable(self.lm_head):
            # Cas tie_embeddings=True
            logits = self.lm_head(x)
        else:
            # Cas avec couche Linear ou BitLinear
            logits = self.lm_head(x)
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 50, 
                temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Génération autoregressive simple pour le Tic-Tac-Toe
        
        Args:
            input_ids: Séquence de départ (batch_size, seq_len)
            max_length: Longueur maximale à générer
            temperature: Température pour le sampling
            top_k: Filtrage top-k optionnel
            
        Returns:
            generated: Séquence générée complète
        """
        self.eval()
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = self(generated)
                
                # Prendre les logits du dernier token
                next_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering si demandé
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_logits, top_k, dim=-1)
                    next_logits = torch.full_like(next_logits, float('-inf'))
                    next_logits.scatter_(-1, top_k_indices, top_k_logits)
                
                # Sampling
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Ajouter à la séquence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Arrêter si on génère le token END
                if next_token.item() == 15:  # END_TOKEN
                    break
                
                # Éviter les séquences trop longues
                if generated.shape[1] >= self.config.max_seq_len:
                    break
        
        return generated
    
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
            
            # Attention stats (si disponible)
            if hasattr(block.attention, 'get_bit_stats'):
                attn_stats = block.attention.get_bit_stats()
                layer_stats['attention'] = attn_stats
                
                if 'qkv_proj' in attn_stats:
                    total_sparsity += attn_stats['qkv_proj'].get('sparsity', 0)
                    total_quantized += attn_stats['qkv_proj'].get('total_params', 0)
            
            # FFN stats (si disponible)
            if hasattr(block.ffn, 'get_bit_stats'):
                ffn_stats = block.ffn.get_bit_stats()
                layer_stats['ffn'] = ffn_stats
                
                if 'total' in ffn_stats:
                    total_sparsity += ffn_stats['total'].get('sparsity', 0)
                    total_quantized += ffn_stats['total'].get('params', 0)
            
            stats['layers'][f'layer_{i}'] = layer_stats
        
        # Stats globales
        embedding_params = self.token_embeddings.weight.numel() + self.pos_embeddings.weight.numel()
        stats['global']['embedding_params'] = embedding_params
        stats['global']['quantized_params'] = total_quantized
        stats['global']['total_params'] = sum(p.numel() for p in self.parameters())
        
        if len(self.blocks) > 0:
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