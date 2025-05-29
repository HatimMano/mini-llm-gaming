# src/models/base_model.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseLLM(nn.Module, ABC):
    """
    Classe de base pour tous les mini-LLMs
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass du modèle"""
        pass
    
    def count_parameters(self) -> int:
        """Compte le nombre total de paramètres"""
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self) -> int:
        """Compte le nombre de paramètres entraînables"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self, use_quantization: bool = True) -> float:
        """
        Calcule la taille du modèle en MB
        Args:
            use_quantization: Si True, utilise 1.58 bits par paramètre
        """
        n_params = self.count_parameters()
        
        if use_quantization:
            # 1.58 bits par paramètre pour quantification ternaire
            bits_per_param = 1.58
        else:
            # 32 bits (float32) par paramètre
            bits_per_param = 32
        
        total_bits = n_params * bits_per_param
        total_bytes = total_bits / 8
        total_mb = total_bytes / (1024 * 1024)
        
        return total_mb
    
    def print_model_info(self):
        """Affiche les informations du modèle"""
        print(f"Modèle: {self.__class__.__name__}")
        print(f"Paramètres totaux: {self.count_parameters():,}")
        print(f"Paramètres entraînables: {self.count_trainable_parameters():,}")
        print(f"Taille non quantifiée: {self.get_model_size_mb(False):.2f} MB")
        print(f"Taille quantifiée (1.58-bit): {self.get_model_size_mb(True):.2f} MB")