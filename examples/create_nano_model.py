# examples/create_nano_model.py
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.nano_config import NanoLLMConfig
from src.models.nano_llm import NanoLLM

def main():
    """Exemple de création et test du NanoLLM"""
    print("=== Création du NanoLLM ===")
    
    # Configuration
    config = NanoLLMConfig()
    
    # Création du modèle
    model = NanoLLM(config)
    model.eval()
    
    # Affichage des informations
    model.print_model_info()
    
    # Test avec des données factices
    print("\n=== Test d'inférence ===")
    batch_size = 2
    seq_len = 32
    
    # Données d'entrée aléatoires
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"Forme d'entrée: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"Forme de sortie: {logits.shape}")
    print(f"Logits min/max: {logits.min():.3f} / {logits.max():.3f}")
    
    # Test de génération simple (next token prediction)
    print("\n=== Test génération token suivant ===")
    probs = torch.softmax(logits[0, -1], dim=-1)
    next_token = torch.multinomial(probs, 1)
    print(f"Token suivant prédit: {next_token.item()}")
    print(f"Probabilité: {probs[next_token].item():.4f}")

if __name__ == "__main__":
    main()