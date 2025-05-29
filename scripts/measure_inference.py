# scripts/measure_inference.py
import torch
import time
import statistics
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.nano_config import NanoLLMConfig
from config.micro_config import MicroLLMConfig
from src.models.nano_llm import NanoLLM
from src.models.micro_llm import MicroLLM

class InferenceBenchmark:
    """
    Benchmark détaillé des performances d'inférence
    """
    def __init__(self):
        self.results = {}
    
    def benchmark_single_forward(self, model, input_ids, n_runs=100, warmup=10):
        """Benchmark un forward pass simple"""
        model.eval()
        
        # Warm-up
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(input_ids)
        
        # Mesures
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = model(input_ids)
                end = time.perf_counter()
                times.append(end - start)
        
        return {
            'mean_time': statistics.mean(times),
            'median_time': statistics.median(times),
            'std_time': statistics.stdev(times) if len(times) > 1 else 0,
            'min_time': min(times),
            'max_time': max(times),
            'times': times
        }
    
    def benchmark_sequence_generation(self, model, start_tokens, max_new_tokens=50):
        """Benchmark génération séquentielle (autoregressive)"""
        model.eval()
        
        batch_size = start_tokens.shape[0]
        generated = start_tokens.clone()
        
        generation_times = []
        
        with torch.no_grad():
            for i in range(max_new_tokens):
                start = time.perf_counter()
                
                # Forward pass
                logits = model(generated)
                
                # Sampling du token suivant
                next_token_logits = logits[:, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, 1)
                
                # Ajout du nouveau token
                generated = torch.cat([generated, next_tokens], dim=1)
                
                end = time.perf_counter()
                generation_times.append(end - start)
        
        total_time = sum(generation_times)
        tokens_generated = batch_size * max_new_tokens
        
        return {
            'total_time': total_time,
            'tokens_per_second': tokens_generated / total_time,
            'time_per_token': total_time / tokens_generated,
            'generation_times': generation_times,
            'generated_sequence': generated
        }
    
    def benchmark_batch_sizes(self, model, vocab_size, seq_len, 
                            batch_sizes=[1, 2, 4, 8, 16]):
        """Benchmark avec différentes tailles de batch"""
        results = {}
        
        for batch_size in batch_sizes:
            try:
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
                result = self.benchmark_single_forward(model, input_ids, n_runs=50)
                
                # Calcul des métriques par batch
                total_tokens = batch_size * seq_len
                tokens_per_second = total_tokens / result['mean_time']
                
                results[batch_size] = {
                    'mean_time': result['mean_time'],
                    'tokens_per_second': tokens_per_second,
                    'throughput_factor': tokens_per_second / batch_sizes[0] if batch_sizes[0] in results or batch_size == batch_sizes[0] else None
                }
                
                if batch_size == batch_sizes[0]:
                    baseline_tps = tokens_per_second
                    results[batch_size]['throughput_factor'] = 1.0
                else:
                    results[batch_size]['throughput_factor'] = tokens_per_second / baseline_tps
                
            except RuntimeError as e:
                results[batch_size] = {'error': str(e)}
        
        return results

def main():
    """Script principal de benchmark"""
    print("=== Benchmark détaillé des performances d'inférence ===")
    
    # Configurations
    nano_config = NanoLLMConfig()
    micro_config = MicroLLMConfig()
    
    # Modèles
    nano_model = NanoLLM(nano_config)
    micro_model = MicroLLM(micro_config)
    
    benchmark = InferenceBenchmark()
    
    # Test 1: Forward pass simple
    print("\n--- Test 1: Forward pass simple ---")
    
    nano_input = torch.randint(0, nano_config.vocab_size, (4, 64))
    micro_input = torch.randint(0, micro_config.vocab_size, (4, 128))
    
    nano_result = benchmark.benchmark_single_forward(nano_model, nano_input)
    micro_result = benchmark.benchmark_single_forward(micro_model, micro_input)
    
    print(f"NanoLLM  - Temps moyen: {nano_result['mean_time']*1000:.2f}ms, "
          f"Tokens/s: {(4*64)/nano_result['mean_time']:.0f}")
    print(f"MicroLLM - Temps moyen: {micro_result['mean_time']*1000:.2f}ms, "
          f"Tokens/s: {(4*128)/micro_result['mean_time']:.0f}")
    
    # Test 2: Génération séquentielle
    print("\n--- Test 2: Génération autoregressive ---")
    
    nano_start = torch.randint(0, nano_config.vocab_size, (2, 10))
    micro_start = torch.randint(0, micro_config.vocab_size, (2, 10))
    
    nano_gen = benchmark.benchmark_sequence_generation(nano_model, nano_start, 30)
    micro_gen = benchmark.benchmark_sequence_generation(micro_model, micro_start, 30)
    
    print(f"NanoLLM  - Génération: {nano_gen['tokens_per_second']:.0f} tokens/s")
    print(f"MicroLLM - Génération: {micro_gen['tokens_per_second']:.0f} tokens/s")
    
    # Test 3: Différentes tailles de batch
    print("\n--- Test 3: Scalabilité batch ---")
    
    nano_batch_results = benchmark.benchmark_batch_sizes(
        nano_model, nano_config.vocab_size, 32, [1, 2, 4, 8]
    )
    
    print("NanoLLM - Scalabilité batch:")
    for batch_size, result in nano_batch_results.items():
        if 'error' not in result:
            print(f"  Batch {batch_size}: {result['tokens_per_second']:.0f} tok/s "
                  f"(facteur: {result['throughput_factor']:.1f}x)")
        else:
            print(f"  Batch {batch_size}: Erreur - {result['error']}")
    
    micro_batch_results = benchmark.benchmark_batch_sizes(
        micro_model, micro_config.vocab_size, 64, [1, 2, 4, 8]
    )
    
    print("MicroLLM - Scalabilité batch:")
    for batch_size, result in micro_batch_results.items():
        if 'error' not in result:
            print(f"  Batch {batch_size}: {result['tokens_per_second']:.0f} tok/s "
                  f"(facteur: {result['throughput_factor']:.1f}x)")
        else:
            print(f"  Batch {batch_size}: Erreur - {result['error']}")

if __name__ == "__main__":
    main()