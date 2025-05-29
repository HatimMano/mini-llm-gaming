# analyze_dataset.py
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List

def analyze_dataset_detailed(filename: str):
    """Analyse détaillée du dataset avec focus sur minimax"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    games = data['games']
    
    # Statistiques par style et position
    stats = defaultdict(lambda: {
        'as_x': {'wins': 0, 'draws': 0, 'losses': 0, 'total': 0},
        'as_o': {'wins': 0, 'draws': 0, 'losses': 0, 'total': 0},
        'vs_styles': defaultdict(lambda: {'wins': 0, 'draws': 0, 'losses': 0, 'total': 0})
    })
    
    # Analyser chaque partie
    for game in games:
        style_x = game['style_x']
        style_o = game['style_o']
        winner = game['winner']
        
        # Résultat pour X
        if winner == 1:  # X gagne
            stats[style_x]['as_x']['wins'] += 1
            stats[style_o]['as_o']['losses'] += 1
            stats[style_x]['vs_styles'][style_o]['wins'] += 1
            stats[style_o]['vs_styles'][style_x]['losses'] += 1
        elif winner == 2:  # O gagne
            stats[style_x]['as_x']['losses'] += 1
            stats[style_o]['as_o']['wins'] += 1
            stats[style_x]['vs_styles'][style_o]['losses'] += 1
            stats[style_o]['vs_styles'][style_x]['wins'] += 1
        else:  # Match nul
            stats[style_x]['as_x']['draws'] += 1
            stats[style_o]['as_o']['draws'] += 1
            stats[style_x]['vs_styles'][style_o]['draws'] += 1
            stats[style_o]['vs_styles'][style_x]['draws'] += 1
        
        stats[style_x]['as_x']['total'] += 1
        stats[style_o]['as_o']['total'] += 1
        stats[style_x]['vs_styles'][style_o]['total'] += 1
        stats[style_o]['vs_styles'][style_x]['total'] += 1
    
    # Afficher les résultats
    print("=== ANALYSE DÉTAILLÉE PAR STYLE ===\n")
    
    for style, data in sorted(stats.items()):
        print(f"\n{style.upper()}:")
        
        # Stats en tant que X
        x_stats = data['as_x']
        if x_stats['total'] > 0:
            x_win_rate = x_stats['wins'] / x_stats['total'] * 100
            x_draw_rate = x_stats['draws'] / x_stats['total'] * 100
            print(f"  En tant que X (premier joueur):")
            print(f"    - Victoires: {x_win_rate:.1f}% ({x_stats['wins']}/{x_stats['total']})")
            print(f"    - Nuls: {x_draw_rate:.1f}% ({x_stats['draws']}/{x_stats['total']})")
            print(f"    - Défaites: {x_stats['losses']}/{x_stats['total']}")
        
        # Stats en tant que O
        o_stats = data['as_o']
        if o_stats['total'] > 0:
            o_win_rate = o_stats['wins'] / o_stats['total'] * 100
            o_draw_rate = o_stats['draws'] / o_stats['total'] * 100
            print(f"  En tant que O (second joueur):")
            print(f"    - Victoires: {o_win_rate:.1f}% ({o_stats['wins']}/{o_stats['total']})")
            print(f"    - Nuls: {o_draw_rate:.1f}% ({o_stats['draws']}/{o_stats['total']})")
            print(f"    - Défaites: {o_stats['losses']}/{o_stats['total']}")
    
    # Analyse spécifique minimax
    print("\n=== FOCUS SUR MINIMAX ===")
    minimax_stats = stats.get('minimax', {})
    
    if minimax_stats:
        print("\nMatchs de minimax contre chaque style:")
        for opponent, results in sorted(minimax_stats['vs_styles'].items()):
            if results['total'] > 0:
                win_rate = results['wins'] / results['total'] * 100
                draw_rate = results['draws'] / results['total'] * 100
                loss_rate = results['losses'] / results['total'] * 100
                print(f"  vs {opponent}:")
                print(f"    - Victoires: {win_rate:.1f}% ({results['wins']}/{results['total']})")
                print(f"    - Nuls: {draw_rate:.1f}% ({results['draws']}/{results['total']})")
                print(f"    - Défaites: {loss_rate:.1f}% ({results['losses']}/{results['total']})")
    
    # Statistiques globales
    print("\n=== STATISTIQUES GLOBALES ===")
    total_games = len(games)
    x_wins = sum(1 for g in games if g['winner'] == 1)
    o_wins = sum(1 for g in games if g['winner'] == 2)
    draws = sum(1 for g in games if g['winner'] == 0)
    
    print(f"Total de parties: {total_games}")
    print(f"Victoires X: {x_wins} ({x_wins/total_games*100:.1f}%)")
    print(f"Victoires O: {o_wins} ({o_wins/total_games*100:.1f}%)")
    print(f"Matchs nuls: {draws} ({draws/total_games*100:.1f}%)")
    
    # Vérifier l'équilibre X/O
    first_x = sum(1 for g in games if g['first_player'] == 1)
    first_o = sum(1 for g in games if g['first_player'] == 2)
    print(f"\nPremier joueur:")
    print(f"X commence: {first_x} ({first_x/total_games*100:.1f}%)")
    print(f"O commence: {first_o} ({first_o/total_games*100:.1f}%)")

def verify_minimax_games(filename: str, num_samples: int = 5):
    """Vérifier quelques parties de minimax pour détecter des bugs"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Trouver des parties minimax vs minimax
    minimax_games = [g for g in data['games'] 
                     if g['style_x'] == 'minimax' and g['style_o'] == 'minimax']
    
    print(f"\n=== VÉRIFICATION MINIMAX vs MINIMAX ===")
    print(f"Nombre de parties minimax vs minimax: {len(minimax_games)}")
    
    if minimax_games:
        # Compter les résultats
        results = {'wins_x': 0, 'wins_o': 0, 'draws': 0}
        for game in minimax_games:
            if game['winner'] == 1:
                results['wins_x'] += 1
            elif game['winner'] == 2:
                results['wins_o'] += 1
            else:
                results['draws'] += 1
        
        print(f"Résultats:")
        print(f"  - X gagne: {results['wins_x']} ({results['wins_x']/len(minimax_games)*100:.1f}%)")
        print(f"  - O gagne: {results['wins_o']} ({results['wins_o']/len(minimax_games)*100:.1f}%)")
        print(f"  - Nuls: {results['draws']} ({results['draws']/len(minimax_games)*100:.1f}%)")
        
        if results['wins_x'] > 0 or results['wins_o'] > 0:
            print("\n⚠️ ALERTE: Minimax ne devrait JAMAIS perdre contre minimax!")
            print("Il y a probablement un bug dans l'implémentation minimax.")
    
    # Vérifier minimax vs random_smart
    print(f"\n=== VÉRIFICATION MINIMAX vs RANDOM_SMART ===")
    minimax_vs_random = [g for g in data['games'] 
                        if (g['style_x'] == 'minimax' and g['style_o'] == 'random_smart') or
                           (g['style_x'] == 'random_smart' and g['style_o'] == 'minimax')]
    
    if minimax_vs_random:
        minimax_wins = 0
        random_wins = 0
        draws = 0
        
        for game in minimax_vs_random:
            if game['winner'] == 0:
                draws += 1
            elif (game['winner'] == 1 and game['style_x'] == 'minimax') or \
                 (game['winner'] == 2 and game['style_o'] == 'minimax'):
                minimax_wins += 1
            else:
                random_wins += 1
        
        print(f"Nombre de parties: {len(minimax_vs_random)}")
        print(f"  - Minimax gagne: {minimax_wins} ({minimax_wins/len(minimax_vs_random)*100:.1f}%)")
        print(f"  - Random gagne: {random_wins} ({random_wins/len(minimax_vs_random)*100:.1f}%)")
        print(f"  - Nuls: {draws} ({draws/len(minimax_vs_random)*100:.1f}%)")
        
        if random_wins > minimax_wins * 0.1:  # Si random gagne plus de 10% du temps de minimax
            print("\n⚠️ ALERTE: Random_smart ne devrait presque jamais battre minimax!")

# Exemple d'utilisation
if __name__ == "__main__":
    analyze_dataset_detailed("ttt_dataset_10k.json")
    verify_minimax_games("ttt_dataset_10k.json")