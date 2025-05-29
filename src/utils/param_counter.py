# src/utils/param_counter.py
import torch
import torch.nn as nn
from typing import Dict, Tuple

def count_parameters_by_layer(model: nn.Module) -> Dict[str, int]:
    """
    Compte les paramètres par couche/composant
    """
    param_count = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Feuille
            n_params = sum(p.numel() for p in module.parameters())
            if n_params > 0:
                param_count[name] = n_params
    
    return param_count

def analyze_model_structure(model: nn.Module) -> Dict:
    """
    Analyse complète de la structure du modèle
    """
    analysis = {
        'total_params': model.count_parameters(),
        'trainable_params': model.count_trainable_parameters(),
        'by_layer': count_parameters_by_layer(model),
        'model_size_mb': {
            'fp32': model.get_model_size_mb(False),
            'quantized': model.get_model_size_mb(True)
        }
    }
    
    # Analyse par type de composant
    component_analysis = {}
    for name, count in analysis['by_layer'].items():
        component_type = name.split('.')[-2] if '.' in name else 'other'
        if component_type not in component_analysis:
            component_analysis[component_type] = 0
        component_analysis[component_type] += count
    
    analysis['by_component'] = component_analysis
    
    return analysis

def print_model_analysis(model: nn.Module, model_name: str = "Model"):
    """
    Affiche une analyse détaillée du modèle
    """
    analysis = analyze_model_structure(model)
    
    print(f"\n=== Analyse de {model_name} ===")
    print(f"Paramètres totaux: {analysis['total_params']:,}")
    print(f"Paramètres entraînables: {analysis['trainable_params']:,}")
    print(f"Taille FP32: {analysis['model_size_mb']['fp32']:.2f} MB")
    print(f"Taille quantifiée: {analysis['model_size_mb']['quantized']:.2f} MB")
    print(f"Compression: {analysis['model_size_mb']['fp32']/analysis['model_size_mb']['quantized']:.1f}x")
    
    print(f"\n--- Répartition par composant ---")
    for component, count in sorted(analysis['by_component'].items(), 
                                 key=lambda x: x[1], reverse=True):
        percentage = (count / analysis['total_params']) * 100
        print(f"{component:15}: {count:>8,} params ({percentage:>5.1f}%)")