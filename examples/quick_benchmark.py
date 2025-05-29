# quick_benchmark.py - Script de test rapide
import torch
import time
import sys
import os
sys.path.append('.')

from config.nano_config import NanoLLMConfig
from config.micro_config import MicroLLMConfig
from src.models.nano_llm import NanoLLM
from src.models.micro_llm import MicroLLM

def benchmark_simple(model, model_name, input_ids, n_runs=50):
    """Benchmark simple et rapide"""
    model.eval()
    
    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_ids)
    
    # Mesures
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            logits = model(input_ids)
            end = time.perf_counter()
            times.append(end - start)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    batch_size, seq_len = input_ids.shape
    total_tokens = batch_size * seq_len
    tokens_per_sec = total_tokens / avg_time
    
    print(f"\n--- {model_name} Performance ---")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Avg time: {avg_time*1000:.2f}ms")
    print(f"Min time: {min_time*1000:.2f}ms") 
    print(f"Max time: {max_time*1000:.2f}ms")
    print(f"Tokens/sec: {tokens_per_sec:.0f}")
    print(f"Throughput: {tokens_per_sec/1000:.1f}K tok/s")
    
    return tokens_per_sec

def memory_usage_estimate(model, model_name):
    """Estimation usage mémoire"""
    print(f"\n--- {model_name} Memory Analysis ---")
    
    # Paramètres
    params = model.count_parameters()
    size_fp32 = model.get_model_size_mb(False)
    size_quantized = model.get_model_size_mb(True)
    
    print(f"Parameters: {params:,}")
    print(f"Size FP32: {size_fp32:.2f} MB")
    print(f"Size 1.58-bit: {size_quantized:.2f} MB")
    print(f"Compression ratio: {size_fp32/size_quantized:.1f}x")
    
    # Estimation instances concurrentes
    available_memory_gb = [4, 8, 16, 32, 64]
    print(f"\nEstimated concurrent instances:")
    for mem_gb in available_memory_gb:
        # Garde 20% de marge + overhead système
        usable_mb = mem_gb * 1024 * 0.7  # 70% utilisable
        instances = int(usable_mb / size_quantized)
        print(f"  {mem_gb}GB RAM: ~{instances:,} instances")

def test_generation_speed(model, model_name, vocab_size, seq_len=10, gen_len=20):
    """Test vitesse de génération autoregressive"""
    print(f"\n--- {model_name} Generation Speed ---")
    
    model.eval()
    start_tokens = torch.randint(0, vocab_size, (1, seq_len))
    generated = start_tokens.clone()
    
    generation_times = []
    
    with torch.no_grad():
        for i in range(gen_len):
            start = time.perf_counter()
            
            logits = model(generated)
            next_token_logits = logits[0, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            end = time.perf_counter()
            generation_times.append(end - start)
    
    total_time = sum(generation_times)
    avg_time_per_token = total_time / gen_len
    tokens_per_sec_gen = 1.0 / avg_time_per_token
    
    print(f"Generated {gen_len} tokens in {total_time:.3f}s")
    print(f"Avg time per token: {avg_time_per_token*1000:.2f}ms")
    print(f"Generation speed: {tokens_per_sec_gen:.1f} tokens/sec")
    print(f"Final sequence length: {generated.shape[1]}")

def main():
    print("=== Quick Benchmark Mini-LLMs ===")
    
    # Créer les modèles
    nano_config = NanoLLMConfig()
    micro_config = MicroLLMConfig()
    
    nano_model = NanoLLM(nano_config)
    micro_model = MicroLLM(micro_config)
    
    # Test d'inférence batch
    print("\n🚀 INFERENCE BENCHMARKS")
    nano_input = torch.randint(0, nano_config.vocab_size, (4, 64))
    micro_input = torch.randint(0, micro_config.vocab_size, (4, 128))
    
    nano_tps = benchmark_simple(nano_model, "NanoLLM", nano_input)
    micro_tps = benchmark_simple(micro_model, "MicroLLM", micro_input)
    
    # Analyse mémoire
    print("\n💾 MEMORY ANALYSIS")
    memory_usage_estimate(nano_model, "NanoLLM")
    memory_usage_estimate(micro_model, "MicroLLM")
    
    # Test génération
    print("\n🎯 GENERATION BENCHMARKS")
    test_generation_speed(nano_model, "NanoLLM", nano_config.vocab_size)
    test_generation_speed(micro_model, "MicroLLM", micro_config.vocab_size)
    
    # Comparaison finale
    print(f"\n📊 SUMMARY COMPARISON")
    print(f"{'Model':<12} {'Params':<10} {'Size':<8} {'Batch TPS':<12} {'Gen TPS':<10}")
    print(f"{'-'*60}")
    print(f"{'NanoLLM':<12} {nano_model.count_parameters()/1e6:.1f}M {nano_model.get_model_size_mb(True):.2f}MB {nano_tps:.0f} {'~50-100':<10}")
    print(f"{'MicroLLM':<12} {micro_model.count_parameters()/1e6:.1f}M {micro_model.get_model_size_mb(True):.2f}MB {micro_tps:.0f} {'~30-60':<10}")
    
    # Ratios
    size_ratio = micro_model.get_model_size_mb(True) / nano_model.get_model_size_mb(True)
    param_ratio = micro_model.count_parameters() / nano_model.count_parameters()
    
    print(f"\nMicroLLM vs NanoLLM ratios:")
    print(f"  Size ratio: {size_ratio:.1f}x larger")
    print(f"  Param ratio: {param_ratio:.1f}x more parameters")
    print(f"  Speed ratio: {nano_tps/micro_tps:.1f}x faster (NanoLLM)")

if __name__ == "__main__":
    main()