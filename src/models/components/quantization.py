# src/models/components/quantization.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BitLinear(nn.Module):
    """
    Couche linéaire quantifiée à 1 bit (ternaire: -1, 0, 1)
    Inspirée de BitNet b1.58
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Poids en pleine précision (seront quantifiés pendant forward)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Pas de biais pour simplifier la quantification
        self.register_parameter('bias', None)
        
        # Initialisation
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialisation optimisée pour la quantification"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def quantize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Quantification ternaire des poids: {-1, 0, 1}
        Utilise la méthode absmean de BitNet
        """
        # Calcul du facteur d'échelle
        scale = torch.mean(torch.abs(weights))
        
        # Quantification ternaire
        weights_scaled = weights / (scale + 1e-8)
        weights_quantized = torch.round(torch.clamp(weights_scaled, -1.0, 1.0))
        
        return weights_quantized * scale
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass avec quantification à la volée"""
        if self.training:
            # En entraînement: quantification "fake" pour gradients
            weight_q = self.quantize_weights(self.weight)
        else:
            # En inférence: quantification réelle
            weight_q = self.quantize_weights(self.weight)
        
        return F.linear(input, weight_q, self.bias)