# train_ttt_1bit.py
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

class TTTDatasetAugmented(Dataset):
    """Dataset TTT avec augmentation par sous-séquences pour modèles 1-bit"""
    def __init__(self, dataset_path: str, max_seq_len: int = 32, train_split: float = 0.9, 
                 enable_subsequences: bool = True, min_subseq_len: int = 8):
        """
        Args:
            dataset_path: Chemin vers le fichier JSON du dataset
            max_seq_len: Longueur maximale des séquences
            train_split: Proportion du dataset pour l'entraînement
            enable_subsequences: Activer l'augmentation par sous-séquences
            min_subseq_len: Longueur minimale des sous-séquences
        """
        # Charger le dataset
        with open(dataset_path, 'r') as f:
            self.data = json.load(f)
        
        self.games = self.data['games']
        self.max_seq_len = max_seq_len
        self.enable_subsequences = enable_subsequences
        self.min_subseq_len = min_subseq_len
        
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
        
        # Générer les sous-séquences pour augmentation
        if enable_subsequences:
            self.train_subsequences = self._generate_subsequences(self.train_games)
            self.val_subsequences = self._generate_subsequences(self.val_games)
        else:
            self.train_subsequences = [(game, 0, len(game['tokens'])) for game in self.train_games]
            self.val_subsequences = [(game, 0, len(game['tokens'])) for game in self.val_games]
        
        print(f"Dataset chargé: {len(self.train_games)} parties train → {len(self.train_subsequences)} échantillons")
        print(f"Validation: {len(self.val_games)} parties → {len(self.val_subsequences)} échantillons")
        
    def _generate_subsequences(self, games: List[Dict]) -> List[Tuple]:
        """Générer des sous-séquences de longueurs variables"""
        subsequences = []
        
        for game in games:
            tokens = game['tokens']
            seq_len = len(tokens)
            
            # Ajouter la séquence complète
            subsequences.append((game, 0, seq_len))
            
            # Générer des sous-séquences de longueurs variables
            if seq_len > self.min_subseq_len:
                # Sous-séquences de 75%, 50% et 25% de la longueur
                for ratio in [0.75, 0.5, 0.25]:
                    subseq_len = max(self.min_subseq_len, int(seq_len * ratio))
                    if subseq_len < seq_len:
                        # Plusieurs positions de départ possibles
                        max_start = min(seq_len - subseq_len, seq_len // 4)
                        for start_pos in range(0, max_start + 1, max(1, max_start // 2)):
                            subsequences.append((game, start_pos, start_pos + subseq_len))
        
        return subsequences
        
    def set_train(self, is_train: bool = True):
        """Basculer entre train et validation"""
        self.is_train = is_train
        
    def __len__(self):
        return len(self.train_subsequences if self.is_train else self.val_subsequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retourne un échantillon avec sous-séquences augmentées
        """
        subsequences = self.train_subsequences if self.is_train else self.val_subsequences
        game, start_idx, end_idx = subsequences[idx]
        
        # Extraire la sous-séquence
        tokens = game['tokens'][start_idx:end_idx]
        
        # Tronquer ou padder à max_seq_len
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        else:
            tokens = tokens + [self.PAD_TOKEN] * (self.max_seq_len - len(tokens))
        
        # Créer input et labels (décalés)
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        # Créer le masque pour les actions
        action_mask = torch.zeros(len(labels), dtype=torch.bool)
        for i in range(len(input_ids)):
            if input_ids[i] == self.ACTION_TOKEN and i < len(labels) - 1:
                action_mask[i + 1] = True
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'action_mask': action_mask,
            'game_id': game['id'],
            'subseq_info': (start_idx, end_idx, len(game['tokens']))
        }

class TTTTrainer1Bit:
    """Entraîneur optimisé pour modèles quantifiés 1-bit"""
    def __init__(self, model, config, dataset_path: str, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Dataset avec augmentation
        self.dataset = TTTDatasetAugmented(
            dataset_path, 
            max_seq_len=config.max_seq_len,
            enable_subsequences=config.enable_subsequences,
            min_subseq_len=config.min_subseq_len
        )
        
        self.train_loader = DataLoader(
            self.dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True  # Important pour stabilité 1-bit
        )
        
        # Optimiseur avec hyperparams spécifiques 1-bit
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,  # Déjà ajusté × 10
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-6  # Plus petit epsilon pour 1-bit
        )
        
        # Scheduler avec warmup étendu
        self.total_steps = len(self.train_loader) * config.num_epochs
        self.warmup_steps = int(config.warmup_ratio * self.total_steps)  # 25% au lieu de 10%
        self.scheduler = self.get_cosine_schedule_with_warmup()
        
        # Loss avec label smoothing pour 1-bit
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.dataset.PAD_TOKEN,
            label_smoothing=config.label_smoothing
        )
        
        # Métriques étendues pour 1-bit
        self.train_losses = []
        self.val_losses = []
        self.action_accuracies = []
        self.legal_move_rates = []
        self.weight_sparsities = []
        self.weight_distributions = []
        self.gradient_norms = []
        
    def get_cosine_schedule_with_warmup(self):
        """Scheduler avec warmup étendu pour 1-bit"""
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))  # Min LR à 10%
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def compute_weight_metrics(self) -> Tuple[float, Dict[str, float]]:
        """Calculer métriques de quantification des poids"""
        total_params = 0
        zero_params = 0
        pos_params = 0
        neg_params = 0
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantize_weights') and hasattr(module, 'weight'):
                # Quantifier les poids pour analyse
                with torch.no_grad():
                    quantized_weights = module.quantize_weights(module.weight)
                    
                total_params += quantized_weights.numel()
                zero_params += (quantized_weights == 0).sum().item()
                pos_params += (quantized_weights > 0).sum().item()
                neg_params += (quantized_weights < 0).sum().item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0
        distribution = {
            'zeros': zero_params / total_params if total_params > 0 else 0,
            'positive': pos_params / total_params if total_params > 0 else 0,
            'negative': neg_params / total_params if total_params > 0 else 0
        }
        
        return sparsity, distribution
    
    def train_epoch(self, epoch: int) -> float:
        """Epoch d'entraînement avec monitoring 1-bit"""
        self.model.train()
        total_loss = 0
        total_action_loss = 0
        correct_actions = 0
        total_actions = 0
        grad_norms = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            action_mask = batch['action_mask'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids)
            
            # Loss totale avec label smoothing
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            
            # Loss spécifique aux actions
            if action_mask.any():
                action_logits = logits[action_mask]
                action_labels = labels[action_mask]
                action_loss = self.criterion(action_logits, action_labels)
                
                # Accuracy des actions
                _, predicted = torch.max(action_logits, dim=-1)
                correct_actions += (predicted == action_labels).sum().item()
                total_actions += action_labels.size(0)
            else:
                action_loss = loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping avec monitoring
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip
            )
            grad_norms.append(total_norm.item())
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Métriques
            total_loss += loss.item()
            total_action_loss += action_loss.item()
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'action_loss': f'{action_loss.item():.4f}',
                'lr': f'{current_lr:.6f}',
                'grad_norm': f'{total_norm:.3f}'
            })
            
            # Logging périodique des métriques 1-bit
            if batch_idx % 100 == 0:
                sparsity, distribution = self.compute_weight_metrics()
                self.weight_sparsities.append(sparsity)
                self.weight_distributions.append(distribution)
        
        avg_loss = total_loss / len(self.train_loader)
        action_accuracy = correct_actions / total_actions if total_actions > 0 else 0
        avg_grad_norm = np.mean(grad_norms)
        self.gradient_norms.append(avg_grad_norm)
        
        # Métriques finales de l'époque
        final_sparsity, final_distribution = self.compute_weight_metrics()
        
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}, Action Acc: {action_accuracy:.2%}")
        print(f"Weight Sparsity: {final_sparsity:.2%}, Avg Grad Norm: {avg_grad_norm:.3f}")
        print(f"Weight Distribution - Zeros: {final_distribution['zeros']:.2%}, "
              f"Pos: {final_distribution['positive']:.2%}, Neg: {final_distribution['negative']:.2%}")
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]:
        """Validation avec métriques 1-bit"""
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
                
                # Vérifier légalité des coups
                legal_moves += self.check_legal_moves(input_ids, predicted, action_mask)
                total_moves += predicted.size(0)
        
        self.dataset.set_train(True)
        
        avg_loss = total_loss / len(val_loader)
        action_accuracy = correct_actions / total_actions if total_actions > 0 else 0
        legal_rate = legal_moves / total_moves if total_moves > 0 else 0
        
        return avg_loss, action_accuracy, legal_rate
    
    def check_legal_moves(self, input_ids: torch.Tensor, predicted_moves: torch.Tensor, 
                         action_mask: torch.Tensor) -> int:
        """Vérifier légalité des coups prédits"""
        legal_count = 0
        batch_size = input_ids.size(0)
        
        for b in range(batch_size):
            tokens = input_ids[b].cpu().numpy()
            pred_idx = 0
            
            for i in range(len(tokens) - 1):
                if action_mask[b, i + 1]:
                    # Trouver l'état du plateau
                    state_start = -1
                    for j in range(i, -1, -1):
                        if tokens[j] == self.dataset.STATE_TOKEN:
                            state_start = j + 1
                            break
                    
                    if state_start != -1 and state_start + 9 <= len(tokens):
                        board_tokens = tokens[state_start:state_start + 9]
                        predicted_pos = predicted_moves[pred_idx].item() - 6
                        
                        if 0 <= predicted_pos < 9:
                            if board_tokens[predicted_pos] == 1:  # EMPTY_TOKEN
                                legal_count += 1
                    
                    pred_idx += 1
        
        return legal_count
    
    def train(self, num_epochs: int, save_dir: str = 'checkpoints_1bit'):
        """Boucle d'entraînement principale pour 1-bit"""
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
                self.save_checkpoint(f'{save_dir}/best_model_1bit.pt', epoch, val_acc)
            
            # Sauvegardes périodiques
            if epoch % 10 == 0:
                self.save_checkpoint(f'{save_dir}/checkpoint_1bit_epoch_{epoch}.pt', epoch, val_acc)
            
            # Graphiques étendus
            if epoch % 5 == 0:
                self.plot_training_curves_1bit(save_dir)
    
    def save_checkpoint(self, path: str, epoch: int, val_acc: float):
        """Sauvegarder checkpoint avec métriques 1-bit"""
        sparsity, distribution = self.compute_weight_metrics()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_accuracy': val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'weight_sparsities': self.weight_sparsities,
            'weight_distributions': self.weight_distributions,
            'gradient_norms': self.gradient_norms,
            'final_sparsity': sparsity,
            'final_distribution': distribution,
            'config': self.config
        }, path)
        print(f"Checkpoint 1-bit sauvegardé: {path}")
    
    def plot_training_curves_1bit(self, save_dir: str):
        """Graphiques étendus pour monitoring 1-bit"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss', alpha=0.8)
        axes[0, 0].plot(self.val_losses, label='Val Loss', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Action accuracy & Legal moves
        axes[0, 1].plot(self.action_accuracies, 'g-', label='Action Accuracy', alpha=0.8)
        axes[0, 1].plot(self.legal_move_rates, 'b-', label='Legal Move Rate', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Rate')
        axes[0, 1].set_title('Action Accuracy & Legal Move Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Weight sparsity
        if self.weight_sparsities:
            axes[1, 0].plot(self.weight_sparsities, 'r-', alpha=0.8)
            axes[1, 0].set_xlabel('Training Steps (×100)')
            axes[1, 0].set_ylabel('Sparsity')
            axes[1, 0].set_title('Weight Sparsity Evolution')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Weight distribution
        if self.weight_distributions:
            zeros = [d['zeros'] for d in self.weight_distributions]
            pos = [d['positive'] for d in self.weight_distributions]
            neg = [d['negative'] for d in self.weight_distributions]
            
            axes[1, 1].plot(zeros, label='Zeros', alpha=0.8)
            axes[1, 1].plot(pos, label='Positive', alpha=0.8)
            axes[1, 1].plot(neg, label='Negative', alpha=0.8)
            axes[1, 1].set_xlabel('Training Steps (×100)')
            axes[1, 1].set_ylabel('Proportion')
            axes[1, 1].set_title('Weight Distribution Evolution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Gradient norms
        if self.gradient_norms:
            axes[2, 0].plot(self.gradient_norms, 'purple', alpha=0.8)
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('Gradient Norm')
            axes[2, 0].set_title('Average Gradient Norm')
            axes[2, 0].grid(True, alpha=0.3)
        
        # Learning rate schedule
        steps = np.arange(self.total_steps)
        lrs = [self.scheduler.lr_lambdas[0](step) * self.config.learning_rate for step in steps]
        axes[2, 1].plot(lrs, 'orange', alpha=0.8)
        axes[2, 1].set_xlabel('Training Steps')
        axes[2, 1].set_ylabel('Learning Rate')
        axes[2, 1].set_title('Learning Rate Schedule')
        axes[2, 1].set_yscale('log')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_curves_1bit.png', dpi=150, bbox_inches='tight')
        plt.close()

@dataclass
class TrainingConfig1Bit:
    """Configuration d'entraînement optimisée pour quantification 1-bit"""
    # Hyperparamètres 1-bit spécifiques
    batch_size: int = 32  # Plus petit batch pour stabilité
    num_epochs: int = 10  # Plus d'époques pour convergence
    learning_rate: float = 3e-3  # LR × 10 pour compenser quantification
    weight_decay: float = 0.005  # Réduit pour éviter sur-régularisation
    gradient_clip: float = 0.5  # Plus agressif pour stabilité
    warmup_ratio: float = 0.25  # Warmup étendu (25% au lieu de 10%)
    label_smoothing: float = 0.1  # Smoothing pour robustesse
    
    # Augmentation de données
    enable_subsequences: bool = True
    min_subseq_len: int = 8
    max_seq_len: int = 32
    
    # Architecture (doit correspondre au modèle)
    vocab_size: int = 16
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    d_ff: int = 256
    d_head: int = 32
    dropout: float = 0.05  # Dropout réduit pour 1-bit
    use_rope: bool = True
    use_squared_relu: bool = True
    tie_embeddings: bool = True
    use_1bit_weights: bool = True  # Flag pour BitLinear

# Script principal
if __name__ == "__main__":
    print("🚀 Lancement de l'entraînement NanoLLM 1-bit pour Tic-Tac-Toe")
    
    # Configuration optimisée 1-bit
    config = TrainingConfig1Bit()
    
    # Créer le modèle avec quantification
    model = NanoLLM(config)
    
    # Vérifier que BitLinear est bien utilisé
    bitlinear_count = sum(1 for name, module in model.named_modules() 
                         if 'BitLinear' in str(type(module)))
    print(f"✅ Modèle créé avec {bitlinear_count} couches BitLinear")
    
    # Calculer la taille théorique du modèle
    total_params = sum(p.numel() for p in model.parameters())
    model_size_1bit = total_params / 8 / 1024 / 1024  # 1 bit par poids → MB
    model_size_fp32 = total_params * 4 / 1024 / 1024  # 32 bits par poids → MB
    
    print(f"📊 Paramètres totaux: {total_params:,}")
    print(f"📦 Taille modèle 1-bit: {model_size_1bit:.2f} MB")
    print(f"📦 Taille modèle FP32: {model_size_fp32:.2f} MB")
    print(f"🎯 Compression: {model_size_fp32/model_size_1bit:.1f}x")
    
    # Créer l'entraîneur
    trainer = TTTTrainer1Bit(model, config, "ttt_dataset_10k.json")
    
    print(f"\n🎮 Configuration d'entraînement:")
    print(f"   • Batch size: {config.batch_size}")
    print(f"   • Learning rate: {config.learning_rate}")
    print(f"   • Epochs: {config.num_epochs}")
    print(f"   • Warmup: {config.warmup_ratio:.1%} des steps")
    print(f"   • Augmentation par sous-séquences: {'✅' if config.enable_subsequences else '❌'}")
    
    # Lancer l'entraînement
    trainer.train(num_epochs=config.num_epochs)
    
    print("🎉 Entraînement terminé !")