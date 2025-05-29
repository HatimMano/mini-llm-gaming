import torch
import psutil
import os
from typing import Dict, List

class MemoryProfiler:
    """
    Profiler mémoire pour analyser l'usage des modèles
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset les métriques"""
        self.measurements = []
        self.baseline_memory = self.get_memory_usage()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Récupère l'usage mémoire actuel"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        usage = {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
        }
        
        # Ajout des métriques PyTorch si disponible
        if torch.cuda.is_available():
            usage['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            usage['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return usage
    
    def checkpoint(self, name: str):
        """Enregistre un point de mesure"""
        current_memory = self.get_memory_usage()
        
        # Calcul des différences par rapport à la baseline
        diff = {}
        for key, value in current_memory.items():
            if key in self.baseline_memory:
                diff[f"delta_{key}"] = value - self.baseline_memory[key]
        
        measurement = {
            'name': name,
            'memory': current_memory,
            'delta': diff
        }
        
        self.measurements.append(measurement)
        return measurement
    
    def profile_model_creation(self, model_class, config, model_name: str):
        """Profile la création d'un modèle"""
        print(f"\n=== Profiling création {model_name} ===")
        
        self.reset()
        self.checkpoint("baseline")
        
        # Création du modèle
        model = model_class(config)
        creation_mem = self.checkpoint("après création")
        
        # Mode eval
        model.eval()
        eval_mem = self.checkpoint("après eval()")
        
        # Forward pass test
        test_input = torch.randint(0, config.vocab_size, (1, 32))
        with torch.no_grad():
            _ = model(test_input)
        forward_mem = self.checkpoint("après forward")
        
        # Affichage des résultats
        print(f"Création modèle: +{creation_mem['delta']['delta_rss_mb']:.2f} MB")
        print(f"Mode eval: +{eval_mem['delta']['delta_rss_mb']:.2f} MB")
        print(f"Forward pass: +{forward_mem['delta']['delta_rss_mb']:.2f} MB")
        print(f"Total utilisé: +{forward_mem['delta']['delta_rss_mb']:.2f} MB")
        
        return model, self.measurements
    
    def estimate_concurrent_instances(self, model_memory_mb: float, 
                                    available_memory_gb: float = 16) -> int:
        """
        Estime le nombre d'instances concurrent possibles
        """
        available_mb = available_memory_gb * 1024
        # Garde 20% de marge de sécurité
        usable_mb = available_mb * 0.8
        
        estimated_instances = int(usable_mb / model_memory_mb)
        
        print(f"\n=== Estimation instances concurrentes ===")
        print(f"Mémoire disponible: {available_memory_gb} GB")
        print(f"Mémoire par modèle: {model_memory_mb:.2f} MB")
        print(f"Instances estimées: {estimated_instances:,}")
        print(f"Utilisation totale: {estimated_instances * model_memory_mb:.0f} MB")
        
        return estimated_instances