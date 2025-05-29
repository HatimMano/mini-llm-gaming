# examples/benchmark_models.py
import torch
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.nano_config import NanoLLMConfig
from config.micro_config import MicroLLMConfig
from src.models.nano_llm import NanoLLM
from src.models.micro_llm import MicroLLM

def benchmark_model(model, model_name, input_ids, n_runs=100):
    """Benchmark un modèle"""
    print(f"\n=== Benchmark {model_name} ===")
    
    model.eval()
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids)
    
    # Mesure du temps
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(n_runs):
            logits = model(input_ids)
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / n_runs
    tokens_per_sec = (input_ids.shape[0] * input_ids.shape[1]) / avg_time
    
    print(f"Temps moyen par forward: {avg_time*1000:.2f}ms")
    print(f"Tokens par seconde: {tokens_per_sec:.0f}")
    
    return avg_time, tokens_per_sec

def main():
    """Compare les performances des deux modèles"""
    print("=== Comparaison des performances NanoLLM vs MicroLLM ===")
    
    # Configurations
    nano_config = NanoLLMConfig()
    micro_config = MicroLLMConfig()
    
    # Modèles
    nano_model = NanoLLM(nano_config)
    micro_model = MicroLLM(micro_config)
    
    # Affichage des infos
    nano_model.print_model_info()
    micro_model.print_model_info()
    
    # Données de test communes
    batch_size = 4
    nano_seq_len = nano_config.max_seq_len // 2  # 128 tokens
    micro_seq_len = micro_config.max_seq_len // 2  # 256 tokens
    
    nano_input = torch.randint(0, nano_config.vocab_size, (batch_size, nano_seq_len))
    micro_input = torch.randint(0, micro_config.vocab_size, (batch_size, micro_seq_len))
    
    # Benchmarks
    nano_time, nano_tps = benchmark_model(nano_model, "NanoLLM", nano_input)
    micro_time, micro_tps = benchmark_model(micro_model, "MicroLLM", micro_input)
    
    # Comparaison
    print(f"\n=== Résumé comparatif ===")
    print(f"NanoLLM  : {nano_model.count_parameters():>8,} params, {nano_model.get_model_size_mb(True):>5.2f}MB, {nano_tps:>6.0f} tok/s")
    print(f"MicroLLM : {micro_model.count_parameters():>8,} params, {micro_model.get_model_size_mb(True):>5.2f}MB, {micro_tps:>6.0f} tok/s")
    
    speedup = nano_time / micro_time if micro_time > 0 else 0
    print(f"Ratio vitesse (Nano/Micro): {speedup:.2f}x")

if __name__ == "__main__":
    main()