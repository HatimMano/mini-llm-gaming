# ttt_testing.py
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import random
from dataclasses import dataclass, field
from functools import partial

# Import des classes nécessaires
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.ttt_dataset_generator import TTTGame, Player, PlayStyle, TTTPlayer, TTTTokenizer
from ttt_training import TTTDataset, TrainingConfig
from src.models.nano_llm import NanoLLM

@dataclass
class TestConfig:
    batch_size: int = 32
    num_test_games: int = 100
    styles_to_test: List[str] = field(default_factory=lambda: ["aggressive", "defensive", "center_first", "corner_first", "random_smart", "minimax"])
    temperature: float = 0.7  # Pour la génération
    top_k: int = 5  # Pour la génération

class TTTTester:
    """Classe pour tester le modèle NanoLLM sur le Tic-Tac-Toe"""
    def __init__(self, model_path: str, dataset_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Charger le modèle
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Charger le dataset pour le tokenizer et les stats
        self.dataset = TTTDataset(dataset_path)
        self.tokenizer = TTTTokenizer()
        
        # Config de test
        self.config = TestConfig()
        
    def load_model(self, model_path: str) -> NanoLLM:
        """Charger le modèle depuis un checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint['config']
        model = NanoLLM(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        return model
    
    def generate_move(self, game: TTTGame, player: Player) -> int:
        """Générer un coup avec le modèle pour le joueur donné"""
        # Convertir l'état actuel en tokens
        history = [(game.board.copy(), None)]  # Pas d'action passée
        tokens = self.tokenizer.tokenize_game(history)
        
        # Tronquer/padder à la longueur max
        if len(tokens) > self.dataset.max_seq_len:
            tokens = tokens[:self.dataset.max_seq_len]
        else:
            tokens = tokens + [self.tokenizer.PAD] * (self.dataset.max_seq_len - len(tokens))
        
        # Convertir en tensor
        input_ids = torch.tensor([tokens[:-1]], dtype=torch.long).to(self.device)
        
        # Génération
        with torch.no_grad():
            logits = self.model(input_ids)[0, -1]  # Dernier token
            
            # Filtrer pour les positions valides seulement
            valid_moves = game.get_valid_moves()
            valid_tokens = [self.tokenizer.action_to_token(move) for move in valid_moves]
            
            # Appliquer temperature et top-k
            logits = logits / self.config.temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Garder seulement les coups valides
            filtered_probs = torch.zeros_like(probs)
            for token in valid_tokens:
                filtered_probs[token] = probs[token]
            
            if self.config.top_k > 0:
                top_k_probs, top_k_indices = torch.topk(filtered_probs, min(self.config.top_k, len(valid_tokens)))
                filtered_probs = torch.zeros_like(probs)
                filtered_probs.scatter_(0, top_k_indices, top_k_probs)
            
            # Échantillonner
            sampled_token = torch.multinomial(filtered_probs, 1).item()
            
            # Convertir le token en position
            if self.tokenizer.POS_0 <= sampled_token <= self.tokenizer.POS_8:
                return sampled_token - self.tokenizer.POS_0
            
        # Fallback: coup aléatoire valide
        return random.choice(valid_moves)
    
    def play_game(self, style: PlayStyle, model_as: Player = Player.X) -> Dict:
        """Jouer une partie contre un style spécifique et retourner les stats"""
        game = TTTGame()
        reference_player = TTTPlayer(style, Player.O if model_as == Player.X else Player.X)
        
        stats = {
            'moves': 0,
            'correct_moves': 0,
            'legal_moves': 0,
            'game_history': []
        }
        
        while not game.is_game_over():
            current_state = game.board.copy()
            
            if game.current_player == model_as:
                # Coup du modèle
                move = self.generate_move(game, game.current_player)
                
                # Vérifier si le coup correspond à ce que ferait le style de référence
                ref_game = TTTGame()
                ref_game.board = current_state.copy()
                ref_game.current_player = game.current_player
                ref_move = reference_player.get_move(ref_game)
                
                stats['correct_moves'] += 1 if move == ref_move else 0
                stats['legal_moves'] += 1 if move in game.get_valid_moves() else 0
                stats['moves'] += 1
                
                stats['game_history'].append({
                    'player': 'model',
                    'move': move,
                    'ref_move': ref_move,
                    'state': current_state
                })
            else:
                # Coup du joueur de référence
                move = reference_player.get_move(game)
                stats['game_history'].append({
                    'player': 'reference',
                    'move': move,
                    'state': current_state
                })
            
            game.make_move(move)
        
        # Résultat final
        winner = game.check_winner()
        stats['winner'] = winner.value if winner else 0
        stats['final_state'] = game.board.copy()
        
        return stats
    
    def test_style(self, style: PlayStyle, num_games: int = 100) -> Dict:
        """Tester le modèle contre un style spécifique sur plusieurs parties"""
        results = {
            'correct_move_rate': [],
            'legal_move_rate': [],
            'win_rate': 0,
            'draw_rate': 0,
            'loss_rate': 0,
            'games': []
        }
        
        for i in tqdm(range(num_games), desc=f"Testing {style.value}"):
            # Alterner qui commence
            model_as = Player.X if i % 2 == 0 else Player.O
            game_stats = self.play_game(style, model_as)
            
            results['games'].append(game_stats)
            results['correct_move_rate'].append(game_stats['correct_moves'] / game_stats['moves'] if game_stats['moves'] > 0 else 0)
            results['legal_move_rate'].append(game_stats['legal_moves'] / game_stats['moves'] if game_stats['moves'] > 0 else 1)
            
            # Calculer les résultats
            if game_stats['winner'] == 0:
                results['draw_rate'] += 1
            elif (game_stats['winner'] == Player.X.value and model_as == Player.X) or \
                 (game_stats['winner'] == Player.O.value and model_as == Player.O):
                results['win_rate'] += 1
            else:
                results['loss_rate'] += 1
        
        # Normaliser les taux
        total = num_games
        results['win_rate'] /= total
        results['draw_rate'] /= total
        results['loss_rate'] /= total
        results['avg_correct'] = np.mean(results['correct_move_rate'])
        results['avg_legal'] = np.mean(results['legal_move_rate'])
        
        return results
    
    def run_tests(self, save_dir: str = 'test_results'):
        """Exécuter tous les tests et sauvegarder les résultats"""
        Path(save_dir).mkdir(exist_ok=True)
        all_results = {}
        
        for style_name in self.config.styles_to_test:
            style = PlayStyle(style_name)
            results = self.test_style(style, self.config.num_test_games)
            all_results[style_name] = results
            
            # Sauvegarder les résultats pour ce style
            with open(f'{save_dir}/{style_name}_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Afficher un résumé
            print(f"\nRésultats pour le style {style_name}:")
            print(f"- Taux de coups corrects: {results['avg_correct']:.2%}")
            print(f"- Taux de coups légaux: {results['avg_legal']:.2%}")
            print(f"- Taux de victoire: {results['win_rate']:.2%}")
            print(f"- Taux de matchs nuls: {results['draw_rate']:.2%}")
            print(f"- Taux de défaite: {results['loss_rate']:.2%}")
        
        # Sauvegarder tous les résultats
        with open(f'{save_dir}/all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Générer des graphiques
        self.plot_results(all_results, save_dir)
        
        return all_results
    
    def plot_results(self, results: Dict, save_dir: str):
        """Générer des graphiques des résultats"""
        styles = list(results.keys())
        
        # Préparer les données
        correct_rates = [results[s]['avg_correct'] for s in styles]
        legal_rates = [results[s]['avg_legal'] for s in styles]
        win_rates = [results[s]['win_rate'] for s in styles]
        draw_rates = [results[s]['draw_rate'] for s in styles]
        
        # Graphique des taux de coups
        plt.figure(figsize=(10, 5))
        x = range(len(styles))
        plt.bar(x, correct_rates, width=0.4, label='Coups corrects')
        plt.bar([i + 0.4 for i in x], legal_rates, width=0.4, label='Coups légaux')
        plt.xticks([i + 0.2 for i in x], styles, rotation=45)
        plt.ylabel('Taux')
        plt.title('Performance du modèle par style')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/move_accuracy.png')
        plt.close()
        
        # Graphique des résultats de jeu
        plt.figure(figsize=(10, 5))
        bottom = np.zeros(len(styles))
        
        for rate, label in zip([win_rates, draw_rates], ['Victoires', 'Matchs nuls']):
            plt.bar(styles, rate, bottom=bottom, label=label)
            bottom += rate
        
        plt.bar(styles, 1 - bottom, bottom=bottom, label='Défaites')
        plt.ylabel('Taux')
        plt.title('Résultats des parties par style')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/game_results.png')
        plt.close()

if __name__ == "__main__":
    # Configuration
    model_path = "checkpoints/best_model.pt"
    dataset_path = "ttt_dataset_10k.json"
    
    # Créer le tester et exécuter les tests
    tester = TTTTester(model_path, dataset_path)
    results = tester.run_tests()
    
    print("\nTests complétés! Résultats sauvegardés dans test_results/")