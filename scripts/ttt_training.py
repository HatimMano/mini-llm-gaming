# train_ttt.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import random
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.nano_llm import NanoLLM
#from config.ttt_config import TTTNanoLLMConfig

class TTTDataset(Dataset):
    """Dataset pour l'entraînement du modèle TTT"""
    def __init__(self, dataset_path: str, max_seq_len: int = 32, train_split: float = 0.9):
        """
        Args:
            dataset_path: Chemin vers le fichier JSON du dataset
            max_seq_len: Longueur maximale des séquences
            train_split: Proportion du dataset pour l'entraînement
        """
        # Charger le dataset
        with open(dataset_path, 'r') as f:
            self.data = json.load(f)
        
        self.games = self.data['games']
        self.max_seq_len = max_seq_len
        
        # Tokens spéciaux
        self.PAD_TOKEN = 0
        self.STATE_TOKEN = 4
        self.ACTION_TOKEN = 5
        self.END_TOKEN = 15
        
        # Séparer train/val
        random.shuffle(self.games)
        split_idx = int(len(self.games) * train_split)
        self.is_train = True
        self.train_games = self.games[:split_idx]
        self.val_games = self.games[split_idx:]
        
        print(f"Dataset chargé: {len(self.train_games)} train, {len(self.val_games)} val")
        
    def set_train(self, is_train: bool = True):
        """Basculer entre train et validation"""
        self.is_train = is_train
        
    def __len__(self):
        return len(self.train_games if self.is_train else self.val_games)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retourne un échantillon avec:
        - input_ids: séquence d'entrée
        - labels: séquence cible (décalée de 1)
        - action_mask: masque pour calculer la loss uniquement sur les actions
        """
        game = (self.train_games if self.is_train else self.val_games)[idx]
        tokens = game['tokens']
        
        # Tronquer ou padder à max_seq_len
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        else:
            tokens = tokens + [self.PAD_TOKEN] * (self.max_seq_len - len(tokens))
        
        # Créer input et labels (décalés)
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        # Créer le masque pour les actions
        # On veut la loss uniquement sur les tokens qui suivent ACTION_TOKEN
        action_mask = torch.zeros(len(labels), dtype=torch.bool)
        for i in range(len(input_ids)):
            if input_ids[i] == self.ACTION_TOKEN and i < len(labels) - 1:
                # Le token suivant ACTION_TOKEN est la position à prédire
                action_mask[i + 1] = True
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'action_mask': action_mask,
            'game_id': game['id']
        }

class TTTTrainer:
    """Entraîneur pour le modèle TTT"""
    def __init__(self, model, config, dataset_path: str, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Dataset et DataLoader
        self.dataset = TTTDataset(dataset_path, max_seq_len=config.max_seq_len)
        self.train_loader = DataLoader(
            self.dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        # Optimiseur
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Scheduler (cosine avec warmup)
        self.total_steps = len(self.train_loader) * config.num_epochs
        self.warmup_steps = int(0.1 * self.total_steps)
        self.scheduler = self.get_cosine_schedule_with_warmup()
        
        # Loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.dataset.PAD_TOKEN)
        
        # Métriques
        self.train_losses = []
        self.val_losses = []
        self.action_accuracies = []
        self.legal_move_rates = []
        
    def get_cosine_schedule_with_warmup(self):
        """Créer un scheduler cosine avec warmup"""
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, epoch: int) -> float:
        """Entraîner pour une époque"""
        self.model.train()
        total_loss = 0
        total_action_loss = 0
        correct_actions = 0
        total_actions = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            # Déplacer vers device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            action_mask = batch['action_mask'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids)
            
            # Calculer la loss totale
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            
            # Calculer la loss sur les actions uniquement
            if action_mask.any():
                action_logits = logits[action_mask]
                action_labels = labels[action_mask]
                action_loss = self.criterion(action_logits, action_labels)
                
                # Calculer l'accuracy sur les actions
                _, predicted = torch.max(action_logits, dim=-1)
                correct_actions += (predicted == action_labels).sum().item()
                total_actions += action_labels.size(0)
            else:
                action_loss = loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Métriques
            total_loss += loss.item()
            total_action_loss += action_loss.item()
            
            # Mise à jour de la barre de progression
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'action_loss': f'{action_loss.item():.4f}',
                'lr': f'{current_lr:.6f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        action_accuracy = correct_actions / total_actions if total_actions > 0 else 0
        
        print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}, Action Accuracy: {action_accuracy:.2%}")
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]:
        """Validation du modèle"""
        self.model.eval()
        self.dataset.set_train(False)
        
        val_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        total_loss = 0
        correct_actions = 0
        total_actions = 0
        legal_moves = 0
        total_moves = 0
        
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            action_mask = batch['action_mask'].to(self.device)
            
            logits = self.model(input_ids)
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            total_loss += loss.item()
            
            if action_mask.any():
                action_logits = logits[action_mask]
                action_labels = labels[action_mask]
                
                _, predicted = torch.max(action_logits, dim=-1)
                correct_actions += (predicted == action_labels).sum().item()
                total_actions += action_labels.size(0)
                
                # Vérifier la légalité des coups
                legal_moves += self.check_legal_moves(input_ids, predicted, action_mask)
                total_moves += predicted.size(0)
        
        self.dataset.set_train(True)
        
        avg_loss = total_loss / len(val_loader)
        action_accuracy = correct_actions / total_actions if total_actions > 0 else 0
        legal_rate = legal_moves / total_moves if total_moves > 0 else 0
        
        return avg_loss, action_accuracy, legal_rate
    
    def check_legal_moves(self, input_ids: torch.Tensor, predicted_moves: torch.Tensor, 
                         action_mask: torch.Tensor) -> int:
        """Vérifier si les coups prédits sont légaux"""
        legal_count = 0
        batch_size = input_ids.size(0)
        
        for b in range(batch_size):
            # Reconstruire l'état du plateau au moment de chaque action
            tokens = input_ids[b].cpu().numpy()
            pred_idx = 0
            
            for i in range(len(tokens) - 1):
                if action_mask[b, i + 1]:  # C'est une position d'action
                    # Trouver le dernier état STATE_TOKEN avant cette action
                    state_start = -1
                    for j in range(i, -1, -1):
                        if tokens[j] == self.dataset.STATE_TOKEN:
                            state_start = j + 1
                            break
                    
                    if state_start != -1 and state_start + 9 <= len(tokens):
                        # Extraire l'état du plateau
                        board_tokens = tokens[state_start:state_start + 9]
                        
                        # Vérifier si la position prédite est vide
                        predicted_pos = predicted_moves[pred_idx].item() - 6  # Tokens 6-14 → positions 0-8
                        
                        if 0 <= predicted_pos < 9:
                            # 1 = EMPTY_TOKEN
                            if board_tokens[predicted_pos] == 1:
                                legal_count += 1
                    
                    pred_idx += 1
        
        return legal_count
    
    def train(self, num_epochs: int, save_dir: str = 'checkpoints'):
        """Boucle d'entraînement principale"""
        Path(save_dir).mkdir(exist_ok=True)
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            # Entraînement
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_acc, legal_rate = self.validate()
            self.val_losses.append(val_loss)
            self.action_accuracies.append(val_acc)
            self.legal_move_rates.append(legal_rate)
            
            print(f"Validation - Loss: {val_loss:.4f}, Action Acc: {val_acc:.2%}, Legal Moves: {legal_rate:.2%}")
            
            # Sauvegarder le meilleur modèle
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(f'{save_dir}/best_model.pt', epoch, val_acc)
            
            # Sauvegarder périodiquement
            if epoch % 10 == 0:
                self.save_checkpoint(f'{save_dir}/checkpoint_epoch_{epoch}.pt', epoch, val_acc)
            
            # Afficher les courbes
            if epoch % 5 == 0:
                self.plot_training_curves(save_dir)
    
    def save_checkpoint(self, path: str, epoch: int, val_acc: float):
        """Sauvegarder un checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_accuracy': val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }, path)
        print(f"Checkpoint sauvegardé: {path}")
    
    def plot_training_curves(self, save_dir: str):
        """Afficher et sauvegarder les courbes d'entraînement"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Action accuracy
        axes[0, 1].plot(self.action_accuracies, 'g-')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Action Prediction Accuracy')
        axes[0, 1].grid(True)
        
        # Legal move rate
        axes[1, 0].plot(self.legal_move_rates, 'b-')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Legal Move Rate')
        axes[1, 0].set_title('Legal Move Rate')
        axes[1, 0].grid(True)
        
        # Learning rate
        lrs = [self.optimizer.param_groups[0]['lr']] * len(self.train_losses)
        axes[1, 1].plot(lrs, 'r-')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_curves.png')
        plt.close()

@dataclass
class TrainingConfig:
    # Hyperparamètres d'entraînement
    batch_size: int = 64
    num_epochs: int = 25
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # Dataset
    max_seq_len: int = 32
    
    # Modèle (à fusionner avec TTTNanoLLMConfig)
    vocab_size: int = 16
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    d_ff: int = 256
    d_head: int = 32
    dropout: float = 0.1  # Pour l'entraînement
    use_rope: bool = True
    use_squared_relu: bool = True
    tie_embeddings: bool = True

# Script d'entraînement principal
if __name__ == "__main__":
    # Configuration
    config = TrainingConfig()
    
    model = NanoLLM(config)
    
    trainer = TTTTrainer(model, config, "ttt_dataset_10k.json")
    
    trainer.train(num_epochs=config.num_epochs)
    
    print("Configuration d'entraînement prête!")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")