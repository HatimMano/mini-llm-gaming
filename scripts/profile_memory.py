# scripts/profile_memory.py
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.nano_config import NanoLLMConfig
from config.micro_config import MicroLLMConfig
from src.models.nano_llm import NanoLLM
from src.models.micro_llm import MicroLLM
from src.utils.memory_profiler import MemoryProfiler
from src.utils.param_counter import print_model_analysis

def main():
    """Script de profilage mémoire complet"""
    print("=== Profilage mémoire des Mini-LLMs ===")
    
    profiler = MemoryProfiler()
    
    # Test NanoLLM
    nano_config = NanoLLMConfig()
    nano_model, nano_measurements = profiler.profile_model_creation(
        NanoLLM, nano_config, "NanoLLM"
    )
    
    # Analyse détaillée NanoLLM
    print_model_analysis(nano_model, "NanoLLM")
    
    # Test MicroLLM
    micro_config = MicroLLMConfig()
    micro_model, micro_measurements = profiler.profile_model_creation(
        MicroLLM, micro_config, "MicroLLM"
    )
    
    # Analyse détaillée MicroLLM
    print_model_analysis(micro_model, "MicroLLM")
    
    # Estimations d'instances concurrentes
    nano_memory = nano_measurements[-1]['delta']['delta_rss_mb']
    micro_memory = micro_measurements[-1]['delta']['delta_rss_mb']
    
    print(f"\n=== Estimations déploiement massif ===")
    
    for memory_gb in [16, 32, 64, 128]:
        print(f"\n--- Serveur {memory_gb}GB ---")
        nano_instances = profiler.estimate_concurrent_instances(nano_memory, memory_gb)
        micro_instances = profiler.estimate_concurrent_instances(micro_memory, memory_gb)
        
        print(f"NanoLLM : {nano_instances:,} instances")
        print(f"MicroLLM: {micro_instances:,} instances")

if __name__ == "__main__":
    main()