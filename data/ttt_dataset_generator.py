# ttt_dataset_generator.py
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from enum import Enum
import json

class Player(Enum):
    EMPTY = 0
    X = 1
    O = 2

class PlayStyle(Enum):
    AGGRESSIVE = "aggressive"      # Priorité: gagner > bloquer > stratégie
    DEFENSIVE = "defensive"        # Priorité: bloquer > gagner > stratégie  
    CENTER_FIRST = "center_first"  # Toujours jouer au centre si possible
    CORNER_FIRST = "corner_first"  # Préfère les coins
    RANDOM_SMART = "random_smart"  # Aléatoire mais évite les coups stupides
    MINIMAX = "minimax"           # Joue parfaitement

class TTTTokenizer:
    """Tokenizer pour le Tic-Tac-Toe"""
    def __init__(self):
        # Définition des tokens
        self.PAD = 0
        self.EMPTY = 1
        self.X = 2
        self.O = 3
        self.STATE = 4
        self.ACTION = 5
        self.POS_0 = 6
        self.POS_1 = 7
        self.POS_2 = 8
        self.POS_3 = 9
        self.POS_4 = 10
        self.POS_5 = 11
        self.POS_6 = 12
        self.POS_7 = 13
        self.POS_8 = 14
        self.END = 15
        
        self.vocab_size = 16
        
    def state_to_tokens(self, board: List[int]) -> List[int]:
        """Convertit un état du plateau en tokens"""
        tokens = []
        for cell in board:
            if cell == Player.EMPTY.value:
                tokens.append(self.EMPTY)
            elif cell == Player.X.value:
                tokens.append(self.X)
            elif cell == Player.O.value:
                tokens.append(self.O)
        return tokens
    
    def action_to_token(self, position: int) -> int:
        """Convertit une position (0-8) en token"""
        return self.POS_0 + position
    
    def tokenize_game(self, game_history: List[Tuple[List[int], Optional[int]]]) -> List[int]:
        """Tokenize une partie complète"""
        tokens = []
        
        for state, action in game_history:
            # Ajouter l'état
            tokens.append(self.STATE)
            tokens.extend(self.state_to_tokens(state))
            
            # Ajouter l'action si elle existe
            if action is not None:
                tokens.append(self.ACTION)
                tokens.append(self.action_to_token(action))
        
        # Ajouter token de fin
        tokens.append(self.END)
        return tokens
    
    def decode_tokens(self, tokens: List[int]) -> str:
        """Décode les tokens pour visualisation"""
        result = []
        i = 0
        while i < len(tokens):
            if tokens[i] == self.STATE:
                result.append("STATE:")
                i += 1
                # Lire les 9 prochains tokens
                board = []
                for j in range(9):
                    if i + j < len(tokens):
                        if tokens[i + j] == self.EMPTY:
                            board.append(".")
                        elif tokens[i + j] == self.X:
                            board.append("X")
                        elif tokens[i + j] == self.O:
                            board.append("O")
                # Afficher en grille 3x3
                for row in range(3):
                    result.append(" ".join(board[row*3:(row+1)*3]))
                i += 9
            elif tokens[i] == self.ACTION:
                i += 1
                if i < len(tokens) and self.POS_0 <= tokens[i] <= self.POS_8:
                    pos = tokens[i] - self.POS_0
                    result.append(f"ACTION: {pos}")
                i += 1
            elif tokens[i] == self.END:
                result.append("END")
                i += 1
            else:
                i += 1
        return "\n".join(result)

class TTTGame:
    """Moteur de jeu Tic-Tac-Toe"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.board = [Player.EMPTY.value] * 9
        self.current_player = Player.X
        
    def get_valid_moves(self) -> List[int]:
        return [i for i in range(9) if self.board[i] == Player.EMPTY.value]
    
    def make_move(self, position: int) -> bool:
        if self.board[position] != Player.EMPTY.value:
            return False
        self.board[position] = self.current_player.value
        self.current_player = Player.O if self.current_player == Player.X else Player.X
        return True
    
    def check_winner(self) -> Optional[Player]:
        # Lignes gagnantes
        wins = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Horizontales
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Verticales
            [0, 4, 8], [2, 4, 6]              # Diagonales
        ]
        
        for line in wins:
            if self.board[line[0]] != Player.EMPTY.value and \
               self.board[line[0]] == self.board[line[1]] == self.board[line[2]]:
                return Player(self.board[line[0]])
        
        return None
    
    def is_draw(self) -> bool:
        return all(cell != Player.EMPTY.value for cell in self.board) and self.check_winner() is None
    
    def is_game_over(self) -> bool:
        return self.check_winner() is not None or self.is_draw()

class TTTPlayer:
    """Joueur avec différents styles"""
    def __init__(self, style: PlayStyle, player: Player):
        self.style = style
        self.player = player
        
    def get_move(self, game: TTTGame) -> int:
        if self.style == PlayStyle.AGGRESSIVE:
            return self._aggressive_move(game)
        elif self.style == PlayStyle.DEFENSIVE:
            return self._defensive_move(game)
        elif self.style == PlayStyle.CENTER_FIRST:
            return self._center_first_move(game)
        elif self.style == PlayStyle.CORNER_FIRST:
            return self._corner_first_move(game)
        elif self.style == PlayStyle.RANDOM_SMART:
            return self._random_smart_move(game)
        elif self.style == PlayStyle.MINIMAX:
            return self._minimax_move(game)
    
    def _find_winning_move(self, game: TTTGame, player: Player) -> Optional[int]:
        """Trouve un coup gagnant pour le joueur spécifié"""
        for move in game.get_valid_moves():
            # Simuler le coup
            game.board[move] = player.value
            winner = game.check_winner()
            game.board[move] = Player.EMPTY.value
            
            if winner == player:
                return move
        return None
    
    def _aggressive_move(self, game: TTTGame) -> int:
        # 1. Chercher un coup gagnant
        winning = self._find_winning_move(game, self.player)
        if winning is not None:
            return winning
        
        # 2. Bloquer l'adversaire
        opponent = Player.O if self.player == Player.X else Player.X
        blocking = self._find_winning_move(game, opponent)
        if blocking is not None:
            return blocking
        
        # 3. Stratégie: centre > coins > bords
        valid = game.get_valid_moves()
        if 4 in valid:
            return 4
        
        corners = [0, 2, 6, 8]
        available_corners = [c for c in corners if c in valid]
        if available_corners:
            return random.choice(available_corners)
        
        return random.choice(valid)
    
    def _defensive_move(self, game: TTTGame) -> int:
        # 1. Bloquer l'adversaire d'abord
        opponent = Player.O if self.player == Player.X else Player.X
        blocking = self._find_winning_move(game, opponent)
        if blocking is not None:
            return blocking
        
        # 2. Chercher un coup gagnant
        winning = self._find_winning_move(game, self.player)
        if winning is not None:
            return winning
        
        # 3. Jouer défensivement (privilégier le centre et les positions qui bloquent)
        valid = game.get_valid_moves()
        if 4 in valid:
            return 4
        
        # Préférer les positions qui contrôlent plusieurs lignes
        edges = [1, 3, 5, 7]
        available_edges = [e for e in edges if e in valid]
        if available_edges:
            return random.choice(available_edges)
        
        return random.choice(valid)
    
    def _center_first_move(self, game: TTTGame) -> int:
        valid = game.get_valid_moves()
        
        # Toujours essayer le centre en premier
        if 4 in valid:
            return 4
        
        # Ensuite logique standard
        winning = self._find_winning_move(game, self.player)
        if winning is not None:
            return winning
        
        opponent = Player.O if self.player == Player.X else Player.X
        blocking = self._find_winning_move(game, opponent)
        if blocking is not None:
            return blocking
        
        return random.choice(valid)
    
    def _corner_first_move(self, game: TTTGame) -> int:
        valid = game.get_valid_moves()
        corners = [0, 2, 6, 8]
        
        # Préférer les coins
        available_corners = [c for c in corners if c in valid]
        if available_corners and len(valid) > 5:  # Début de partie
            return random.choice(available_corners)
        
        # Logique standard
        winning = self._find_winning_move(game, self.player)
        if winning is not None:
            return winning
        
        opponent = Player.O if self.player == Player.X else Player.X
        blocking = self._find_winning_move(game, opponent)
        if blocking is not None:
            return blocking
        
        if available_corners:
            return random.choice(available_corners)
        
        return random.choice(valid)
    
    def _random_smart_move(self, game: TTTGame) -> int:
        # Toujours bloquer/gagner si possible
        winning = self._find_winning_move(game, self.player)
        if winning is not None:
            return winning
        
        opponent = Player.O if self.player == Player.X else Player.X
        blocking = self._find_winning_move(game, opponent)
        if blocking is not None:
            return blocking
        
        # Sinon, coup aléatoire
        return random.choice(game.get_valid_moves())
    
    def _minimax_move(self, game: TTTGame) -> int:
        """Implémentation minimax pour jeu parfait"""
        _, best_move = self._minimax(game, self.player, True, -float('inf'), float('inf'))
        return best_move
    
    def _minimax(self, game: TTTGame, player: Player, maximizing: bool, alpha: float, beta: float) -> Tuple[float, Optional[int]]:
        # État terminal
        winner = game.check_winner()
        if winner == self.player:
            return 1, None
        elif winner is not None:
            return -1, None
        elif game.is_draw():
            return 0, None
        
        best_move = None
        
        if maximizing:
            max_eval = -float('inf')
            for move in game.get_valid_moves():
                game.board[move] = player.value
                next_player = Player.O if player == Player.X else Player.X
                eval_score, _ = self._minimax(game, next_player, False, alpha, beta)
                game.board[move] = Player.EMPTY.value
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in game.get_valid_moves():
                game.board[move] = player.value
                next_player = Player.O if player == Player.X else Player.X
                eval_score, _ = self._minimax(game, next_player, True, alpha, beta)
                game.board[move] = Player.EMPTY.value
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return min_eval, best_move

def generate_game(style1: PlayStyle, style2: PlayStyle, first_player: Player = Player.X) -> List[Tuple[List[int], Optional[int]]]:
    """Génère une partie complète entre deux styles"""
    game = TTTGame()
    
    # Déterminer qui joue X et O selon first_player
    if first_player == Player.X:
        player_x = TTTPlayer(style1, Player.X)
        player_o = TTTPlayer(style2, Player.O)
    else:
        player_x = TTTPlayer(style2, Player.X)
        player_o = TTTPlayer(style1, Player.O)
    
    history = []
    
    while not game.is_game_over():
        # Sauvegarder l'état actuel
        current_state = game.board.copy()
        
        # Obtenir le coup du joueur actuel
        if game.current_player == Player.X:
            move = player_x.get_move(game)
        else:
            move = player_o.get_move(game)
        
        # Ajouter à l'historique
        history.append((current_state, move))
        
        # Jouer le coup
        game.make_move(move)
    
    # Ajouter l'état final sans action
    history.append((game.board.copy(), None))
    
    return history

def generate_dataset(num_games: int, style_distribution: Optional[Dict[PlayStyle, float]] = None) -> Dict:
    """
    Génère un dataset complet avec différents styles
    
    Args:
        num_games: Nombre total de parties à générer
        style_distribution: Distribution des styles (optionnel)
    """
    if style_distribution is None:
        # Distribution par défaut équilibrée
        styles = list(PlayStyle)
        style_distribution = {style: 1.0/len(styles) for style in styles}
    
    tokenizer = TTTTokenizer()
    dataset = {
        'games': [],
        'metadata': {
            'num_games': num_games,
            'vocab_size': tokenizer.vocab_size,
            'styles': [s.value for s in PlayStyle],
            'style_distribution': {s.value: p for s, p in style_distribution.items()}
        }
    }
    
    # Normaliser la distribution
    total = sum(style_distribution.values())
    style_distribution = {s: p/total for s, p in style_distribution.items()}
    
    # Générer les parties
    for i in range(num_games):
        # Choisir les styles selon la distribution
        styles = list(style_distribution.keys())
        probs = list(style_distribution.values())
        
        style1 = np.random.choice(styles, p=probs)
        style2 = np.random.choice(styles, p=probs)
        
        # Alterner le premier joueur
        first_player = Player.X if i % 2 == 0 else Player.O
        
        # Générer la partie
        game_history = generate_game(style1, style2, first_player)
        
        # Tokenizer la partie
        tokens = tokenizer.tokenize_game(game_history)
        
        # Déterminer le gagnant
        final_board = game_history[-1][0]
        game = TTTGame()
        game.board = final_board
        winner = game.check_winner()
        
        # Ajouter au dataset
        dataset['games'].append({
            'id': i,
            'tokens': tokens,
            'style_x': style1.value if first_player == Player.X else style2.value,
            'style_o': style2.value if first_player == Player.X else style1.value,
            'first_player': first_player.value,
            'winner': winner.value if winner else 0,  # 0 pour match nul
            'num_moves': len([h for h in game_history if h[1] is not None])
        })
        
        # Affichage de progression
        if (i + 1) % 1000 == 0:
            print(f"Généré {i + 1}/{num_games} parties...")
    
    return dataset

def save_dataset(dataset: Dict, filename: str):
    """Sauvegarde le dataset dans un fichier JSON"""
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset sauvegardé dans {filename}")

def load_dataset(filename: str) -> Dict:
    """Charge un dataset depuis un fichier JSON"""
    with open(filename, 'r') as f:
        return json.load(f)

# Exemple d'utilisation
if __name__ == "__main__":
    # Générer un dataset avec distribution personnalisée
    style_dist = {
        PlayStyle.AGGRESSIVE: 0.2,
        PlayStyle.DEFENSIVE: 0.2,
        PlayStyle.CENTER_FIRST: 0.15,
        PlayStyle.CORNER_FIRST: 0.15,
        PlayStyle.RANDOM_SMART: 0.2,
        PlayStyle.MINIMAX: 0.1  # Moins de parties parfaites
    }
    
    # Générer 10,000 parties
    print("Génération du dataset...")
    dataset = generate_dataset(200000, style_dist)
    
    # Sauvegarder
    save_dataset(dataset, "ttt_dataset_10k.json")
    
    # Statistiques
    print(f"\nStatistiques du dataset:")
    print(f"- Nombre de parties: {len(dataset['games'])}")
    
    # Compter les victoires par style
    style_wins = {style: {'wins': 0, 'games': 0} for style in PlayStyle}
    draws = 0
    
    for game in dataset['games']:
        if game['winner'] == 0:
            draws += 1
        elif game['winner'] == Player.X.value:
            style = PlayStyle(game['style_x'])
            style_wins[style]['wins'] += 1
        else:
            style = PlayStyle(game['style_o'])
            style_wins[style]['wins'] += 1
        
        style_wins[PlayStyle(game['style_x'])]['games'] += 1
        style_wins[PlayStyle(game['style_o'])]['games'] += 1
    
    print(f"- Matchs nuls: {draws} ({draws/len(dataset['games'])*100:.1f}%)")
    print("\nTaux de victoire par style:")
    for style, stats in style_wins.items():
        if stats['games'] > 0:
            win_rate = stats['wins'] / stats['games'] * 100
            print(f"  - {style.value}: {win_rate:.1f}% ({stats['wins']}/{stats['games']} parties)")
    
    # Exemple de décodage
    print("\nExemple de partie tokenisée:")
    tokenizer = TTTTokenizer()
    example_game = dataset['games'][0]
    print(f"Styles: {example_game['style_x']} (X) vs {example_game['style_o']} (O)")
    print(f"Premier joueur: {'X' if example_game['first_player'] == Player.X.value else 'O'}")
    print(f"Tokens: {example_game['tokens'][:30]}...")
    print("\nDécodage:")
    print(tokenizer.decode_tokens(example_game['tokens']))