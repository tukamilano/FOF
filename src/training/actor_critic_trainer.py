"""
Actor-Critic Training Functions
PPOアルゴリズムとKL制約を組み合わせた学習関数を実装
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from tqdm import tqdm

from ..core.actor_critic_model import ActorCriticModel


class RLDataset(Dataset):
    """Reinforcement Learning用のデータセット"""
    
    def __init__(
        self,
        successful_tactics: List[Dict[str, Any]],
        failed_tactics: List[Dict[str, Any]],
        tokenizer,
        max_seq_len: int = 256
    ):
        self.data = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # 成功データを追加
        for tactic_data in successful_tactics:
            self.data.append({
                **tactic_data,
                'is_success': True
            })
        
        # 失敗データを追加
        for tactic_data in failed_tactics:
            self.data.append({
                **tactic_data,
                'is_success': False
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            if idx >= len(self.data):
                print(f"⚠️  Index {idx} out of range (dataset size: {len(self.data)})")
                return self._get_dummy_item()
            
            item = self.data[idx]
            if item is None:
                print(f"⚠️  Item at index {idx} is None")
                return self._get_dummy_item()
            
            # 入力をエンコード
            premises = item.get('premises', [])
            goal = item.get('goal', '')
            
            # premisesがNoneでないことをチェック（空のリストはOK）
            if premises is None or not goal:
                print(f"⚠️  Missing premises or goal at index {idx}")
                return self._get_dummy_item()
            
            input_ids, attention_mask, segment_ids = self.tokenizer.encode(
                goal, premises, self.max_seq_len
            )
            
            # タクティクを解析
            tactic = item.get('tactic', {})
            if isinstance(tactic, str):
                from ..core.state_encoder import parse_tactic_string
                tactic_dict = parse_tactic_string(tactic)
            else:
                tactic_dict = tactic
            
            # 戦術名を数値IDに変換（簡易的なマッピング）
            main_tactic = tactic_dict.get('main', 'assumption')
            arg1_value = tactic_dict.get('arg1', '0')
            arg2_value = tactic_dict.get('arg2', '0')
            
            # None値の処理
            if arg1_value is None:
                arg1_value = '0'
            if arg2_value is None:
                arg2_value = '0'
            
            # 文字列に変換
            arg1_value = str(arg1_value)
            arg2_value = str(arg2_value)
            
            # 基本的な戦術マッピング
            tactic_mapping = {
                'assumption': 0, 'intro': 1, 'split': 2, 'left': 3, 'right': 4, 'add_dn': 5,
                'apply': 6, 'destruct': 12, 'specialize': 18
            }
            
            # main_actionの計算
            if main_tactic in ['apply', 'destruct', 'specialize']:
                # apply, destruct, specializeの場合は基本ID + 引数
                if main_tactic == 'apply':
                    main_action = 6 + int(arg1_value) if arg1_value.isdigit() else 6
                elif main_tactic == 'destruct':
                    main_action = 12 + int(arg1_value) if arg1_value.isdigit() else 12
                elif main_tactic == 'specialize':
                    main_action = 18 + int(arg1_value) if arg1_value.isdigit() else 18
            else:
                main_action = tactic_mapping.get(main_tactic, 0)
            
            # arg1_action, arg2_actionの計算
            arg1_action = int(arg1_value) if arg1_value.isdigit() else 0
            arg2_action = int(arg2_value) if arg2_value.isdigit() else 0
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'segment_ids': segment_ids,
                'main_action': main_action,
                'arg1_action': arg1_action,
                'arg2_action': arg2_action,
                'reward': float(item.get('reward', 0.0)),
                'log_prob': float(item.get('log_prob', 0.0)),
                'is_success': bool(item.get('is_success', False))
            }
        except Exception as e:
            print(f"⚠️  Error in RLDataset.__getitem__({idx}): {e}")
            print(f"   Item keys: {list(item.keys()) if 'item' in locals() and item is not None else 'N/A'}")
            return self._get_dummy_item()
    
    def _get_dummy_item(self):
        """ダミーデータを返す"""
        try:
            dummy_input_ids = torch.zeros(self.max_seq_len, dtype=torch.long)
            dummy_attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
            dummy_segment_ids = torch.zeros(self.max_seq_len, dtype=torch.long)
            
            result = {
                'input_ids': dummy_input_ids,
                'attention_mask': dummy_attention_mask,
                'segment_ids': dummy_segment_ids,
                'main_action': 0,
                'arg1_action': 0,
                'arg2_action': 0,
                'reward': 0.0,
                'log_prob': 0.0,
                'is_success': False
            }
            
            # 結果がNoneでないことを確認
            if result is None:
                print("⚠️  _get_dummy_item returned None!")
                return self._get_fallback_item()
            
            return result
        except Exception as e:
            print(f"⚠️  Error in _get_dummy_item: {e}")
            return self._get_fallback_item()
    
    def _get_fallback_item(self):
        """最終的なフォールバック用のダミーデータ"""
        return {
            'input_ids': torch.tensor([0] * self.max_seq_len, dtype=torch.long),
            'attention_mask': torch.tensor([0] * self.max_seq_len, dtype=torch.long),
            'segment_ids': torch.tensor([0] * self.max_seq_len, dtype=torch.long),
            'main_action': 0,
            'arg1_action': 0,
            'arg2_action': 0,
            'reward': 0.0,
            'log_prob': 0.0,
            'is_success': False
        }


def compute_ppo_loss(
    actor_logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    old_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    PPO損失を計算
    
    Args:
        actor_logits: (main_logits, arg1_logits, arg2_logits)
        old_log_probs: 古い対数確率
        rewards: 報酬
        values: 価値関数の出力
        advantages: アドバンテージ
        clip_ratio: PPOクリッピング比率
        value_coef: 価値損失の係数
        entropy_coef: エントロピー損失の係数
    
    Returns:
        (total_loss, loss_dict)
    """
    main_logits, arg1_logits, arg2_logits = actor_logits
    
    # 現在の対数確率を計算
    main_log_probs = F.log_softmax(main_logits, dim=-1)
    arg1_log_probs = F.log_softmax(arg1_logits, dim=-1)
    arg2_log_probs = F.log_softmax(arg2_logits, dim=-1)
    
    # アクションを取得（バッチの最初の要素を使用）
    main_actions = torch.argmax(main_log_probs, dim=-1)
    arg1_actions = torch.argmax(arg1_log_probs, dim=-1)
    arg2_actions = torch.argmax(arg2_log_probs, dim=-1)
    
    # 現在の対数確率
    current_log_probs = (
        main_log_probs.gather(1, main_actions.unsqueeze(-1)).squeeze(-1) +
        arg1_log_probs.gather(1, arg1_actions.unsqueeze(-1)).squeeze(-1) +
        arg2_log_probs.gather(1, arg2_actions.unsqueeze(-1)).squeeze(-1)
    )
    
    # 確率比を計算
    ratio = torch.exp(current_log_probs - old_log_probs)
    
    # PPOクリッピング損失
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    
    # 価値関数損失
    value_loss = F.mse_loss(values, rewards)
    
    # エントロピー損失
    main_entropy = -(F.softmax(main_logits, dim=-1) * F.log_softmax(main_logits, dim=-1)).sum(dim=-1).mean()
    arg1_entropy = -(F.softmax(arg1_logits, dim=-1) * F.log_softmax(arg1_logits, dim=-1)).sum(dim=-1).mean()
    arg2_entropy = -(F.softmax(arg2_logits, dim=-1) * F.log_softmax(arg2_logits, dim=-1)).sum(dim=-1).mean()
    entropy_loss = -(main_entropy + arg1_entropy + arg2_entropy)
    
    # 総損失
    total_loss = actor_loss + value_coef * value_loss + entropy_coef * entropy_loss
    
    loss_dict = {
        'actor_loss': actor_loss.item(),
        'value_loss': value_loss.item(),
        'entropy_loss': entropy_loss.item(),
        'total_loss': total_loss.item()
    }
    
    return total_loss, loss_dict


def compute_advantages(
    rewards: List[float],
    values: List[float],
    gamma: float = 0.99,
    lam: float = 0.95
) -> List[float]:
    """
    GAE (Generalized Advantage Estimation) でアドバンテージを計算
    
    Args:
        rewards: 報酬のリスト
        values: 価値関数の出力のリスト
        gamma: 割引率
        lam: GAEパラメータ
    
    Returns:
        アドバンテージのリスト
    """
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    return advantages


def train_actor_critic_epoch(
    model: ActorCriticModel,
    dataloader: DataLoader,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    device: torch.device,
    kl_penalty_weight: float = 0.1,
    entropy_weight: float = 0.01,
    ppo_epochs: int = 4,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    gamma: float = 0.99,
    lam: float = 0.95,
    use_amp: bool = False,
    scaler: Optional[GradScaler] = None,
    use_wandb: bool = False,
    epoch: int = 0,
    log_frequency: int = 100
) -> Dict[str, float]:
    """
    Actor-Critic学習の1エポックを実行
    
    Args:
        model: Actor-Criticモデル
        dataloader: データローダー
        actor_optimizer: Actor用オプティマイザー
        critic_optimizer: Critic用オプティマイザー
        device: デバイス
        kl_penalty_weight: KL制約の重み
        entropy_weight: エントロピー正則化の重み
        ppo_epochs: PPOの更新回数
        clip_ratio: PPOクリッピング比率
        value_coef: 価値損失の係数
        gamma: 割引率
        lam: GAEパラメータ
        use_amp: 混合精度を使用するか
        scaler: 混合精度スケーラー
        use_wandb: wandbを使用するか
        epoch: エポック番号
        log_frequency: ログ頻度
    
    Returns:
        損失の辞書
    """
    model.train()
    
    total_losses = {
        'actor_loss': 0.0,
        'critic_loss': 0.0,
        'kl_loss': 0.0,
        'entropy_loss': 0.0,
        'total_loss': 0.0
    }
    
    num_batches = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training Actor-Critic")):
        # バッチデータを取得
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        main_actions = batch['main_action'].to(device)
        arg1_actions = batch['arg1_action'].to(device)
        arg2_actions = batch['arg2_action'].to(device)
        rewards = batch['reward'].to(device).float()  # float32に変換
        old_log_probs = batch['log_prob'].to(device).float()  # float32に変換
        
        # エピソードごとにアドバンテージを計算
        # ここでは簡略化して、即座の報酬を使用
        advantages = rewards - rewards.mean()
        
        # PPOの複数回更新
        for ppo_epoch in range(ppo_epochs):
            # Actor推論
            if use_amp and scaler is not None:
                with autocast():
                    main_logits, arg1_logits, arg2_logits, values, pretrained_logits = model(
                        input_ids, attention_mask, segment_ids, return_pretrained_logits=True
                    )
                    
                    # PPO損失を計算
                    ppo_loss, ppo_loss_dict = compute_ppo_loss(
                        (main_logits, arg1_logits, arg2_logits),
                        old_log_probs,
                        rewards,
                        values,
                        advantages,
                        clip_ratio,
                        value_coef,
                        entropy_weight
                    )
                    
                    # KL制約を計算
                    kl_loss = torch.tensor(0.0, device=device)
                    if pretrained_logits is not None:
                        kl_loss = model.compute_kl_divergence(
                            (main_logits, arg1_logits, arg2_logits),
                            pretrained_logits
                        )
                    
                    # 総損失
                    total_loss = ppo_loss + kl_penalty_weight * kl_loss
                
                # 逆伝播
                scaler.scale(total_loss).backward()
                scaler.step(actor_optimizer)
                scaler.step(critic_optimizer)
                scaler.update()
                
            else:
                # 通常の推論
                main_logits, arg1_logits, arg2_logits, values, pretrained_logits = model(
                    input_ids, attention_mask, segment_ids, return_pretrained_logits=True
                )
                
                # PPO損失を計算
                ppo_loss, ppo_loss_dict = compute_ppo_loss(
                    (main_logits, arg1_logits, arg2_logits),
                    old_log_probs,
                    rewards,
                    values,
                    advantages,
                    clip_ratio,
                    value_coef,
                    entropy_weight
                )
                
                # KL制約を計算
                kl_loss = torch.tensor(0.0, device=device)
                if pretrained_logits is not None:
                    kl_loss = model.compute_kl_divergence(
                        (main_logits, arg1_logits, arg2_logits),
                        pretrained_logits
                    )
                
                # 総損失
                total_loss = ppo_loss + kl_penalty_weight * kl_loss
                
                # 逆伝播
                total_loss.backward()
                actor_optimizer.step()
                critic_optimizer.step()
            
            # オプティマイザーをリセット
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            
            # 損失を累積
            total_losses['actor_loss'] += ppo_loss_dict['actor_loss']
            total_losses['critic_loss'] += ppo_loss_dict['value_loss']
            total_losses['kl_loss'] += kl_loss.item()
            total_losses['entropy_loss'] += ppo_loss_dict['entropy_loss']
            total_losses['total_loss'] += total_loss.item()
        
        num_batches += 1
        
        # ログ出力
        if batch_idx % log_frequency == 0:
            avg_losses = {k: v / num_batches for k, v in total_losses.items()}
            print(f"Epoch {epoch}, Batch {batch_idx}: {avg_losses}")
            
            if use_wandb:
                import wandb
                wandb.log({
                    f"train/{k}": v for k, v in avg_losses.items()
                })
    
    # 平均損失を計算
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    return avg_losses


def train_actor_critic(
    model: ActorCriticModel,
    successful_tactics: List[Dict[str, Any]],
    failed_tactics: List[Dict[str, Any]],
    tokenizer,
    device: torch.device,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    kl_penalty_weight: float = 0.1,
    entropy_weight: float = 0.01,
    ppo_epochs: int = 4,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    gamma: float = 0.99,
    lam: float = 0.95,
    use_amp: bool = False,
    use_wandb: bool = False,
    max_seq_len: int = 256
) -> Dict[str, List[float]]:
    """
    Actor-Critic学習のメイン関数
    
    Args:
        model: Actor-Criticモデル
        successful_tactics: 成功データ
        failed_tactics: 失敗データ
        tokenizer: トークナイザー
        device: デバイス
        num_epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        kl_penalty_weight: KL制約の重み
        entropy_weight: エントロピー正則化の重み
        ppo_epochs: PPOの更新回数
        clip_ratio: PPOクリッピング比率
        value_coef: 価値損失の係数
        gamma: 割引率
        lam: GAEパラメータ
        use_amp: 混合精度を使用するか
        use_wandb: wandbを使用するか
        max_seq_len: 最大シーケンス長
    
    Returns:
        学習履歴の辞書
    """
    # データセットとデータローダーを作成
    dataset = RLDataset(successful_tactics, failed_tactics, tokenizer, max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # オプティマイザーを作成
    actor_optimizer = torch.optim.AdamW(model.shared_encoder.parameters(), lr=learning_rate)
    critic_optimizer = torch.optim.AdamW(model.critic.parameters(), lr=learning_rate)
    
    # 混合精度スケーラー
    scaler = None
    if use_amp and device.type == 'cuda':
        scaler = GradScaler()
    
    # 学習履歴
    history = {
        'actor_loss': [],
        'critic_loss': [],
        'kl_loss': [],
        'entropy_loss': [],
        'total_loss': []
    }
    
    print(f"Starting Actor-Critic training for {num_epochs} epochs...")
    print(f"Dataset size: {len(dataset)} (successful: {len(successful_tactics)}, failed: {len(failed_tactics)})")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # 1エポックの学習を実行
        epoch_losses = train_actor_critic_epoch(
            model=model,
            dataloader=dataloader,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            device=device,
            kl_penalty_weight=kl_penalty_weight,
            entropy_weight=entropy_weight,
            ppo_epochs=ppo_epochs,
            clip_ratio=clip_ratio,
            value_coef=value_coef,
            gamma=gamma,
            lam=lam,
            use_amp=use_amp,
            scaler=scaler,
            use_wandb=use_wandb,
            epoch=epoch
        )
        
        # 履歴に追加
        for key, value in epoch_losses.items():
            history[key].append(value)
        
        print(f"Epoch {epoch + 1} completed: {epoch_losses}")
    
    print("Actor-Critic training completed!")
    
    return history
