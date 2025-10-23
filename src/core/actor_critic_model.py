"""
Actor-Critic Model for Reinforcement Learning
既存のTransformerClassifierをベースにActor-Criticアーキテクチャを実装
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np

from .transformer_classifier import TransformerClassifier


class ActorCriticModel(nn.Module):
    """
    Actor-Critic Model for tactic selection in theorem proving
    
    Architecture:
    - Shared Encoder: Transformer-based encoder (from pretrained model)
    - Actor Head: 3 hierarchical heads (main, arg1, arg2) for action selection
    - Critic Head: Value function for state evaluation
    - Reference Model: Frozen pretrained model for KL regularization
    """
    
    def __init__(
        self,
        base_transformer: TransformerClassifier,
        pretrained_model: TransformerClassifier,
        critic_hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 共有エンコーダー（base_transformerの重みを使用）
        self.shared_encoder = base_transformer
        
        # Actor head（既存の3つのヘッドを活用）
        self.actor_main = base_transformer.head_main
        self.actor_arg1 = base_transformer.head_arg1
        self.actor_arg2 = base_transformer.head_arg2
        
        # Critic head（新規追加）
        encoder_dim = base_transformer.encoder.layers[0].self_attn.embed_dim
        self.critic = nn.Sequential(
            nn.Linear(encoder_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(critic_hidden_dim, critic_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(critic_hidden_dim // 2, 1)
        )
        
        # 参照モデル（pretrained_model、凍結）
        self.pretrained_model = pretrained_model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        
        # モデル情報を保存
        self.vocab_size = base_transformer.vocab_size
        self.pad_id = base_transformer.pad_id
        self.max_seq_len = base_transformer.max_seq_len
        self.num_main_classes = base_transformer.num_main_classes
        self.num_arg1_classes = base_transformer.num_arg1_classes
        self.num_arg2_classes = base_transformer.num_arg2_classes
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        return_pretrained_logits: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of Actor-Critic model
        
        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            segment_ids: Segment IDs (batch, seq_len)
            return_pretrained_logits: Whether to return pretrained model logits for KL loss
        
        Returns:
            (actor_main_logits, actor_arg1_logits, actor_arg2_logits, critic_value, pretrained_logits)
        """
        # 共有エンコーダーで特徴量を抽出
        shared_features = self._encode_shared(input_ids, attention_mask, segment_ids)
        
        # Actor headでアクション確率を計算
        actor_main_logits = self.actor_main(shared_features)
        actor_arg1_logits = self.actor_arg1(shared_features)
        actor_arg2_logits = self.actor_arg2(shared_features)
        
        # Critic headで価値を計算
        critic_value = self.critic(shared_features).squeeze(-1)  # (batch,)
        
        # 必要に応じてpretrained modelのlogitsも計算
        pretrained_logits = None
        if return_pretrained_logits:
            with torch.no_grad():
                pretrained_main, pretrained_arg1, pretrained_arg2 = self.pretrained_model(
                    input_ids, attention_mask, segment_ids
                )
                pretrained_logits = (pretrained_main, pretrained_arg1, pretrained_arg2)
        
        return actor_main_logits, actor_arg1_logits, actor_arg2_logits, critic_value, pretrained_logits
    
    def _encode_shared(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """共有エンコーダーで特徴量を抽出"""
        # 埋め込み
        x = self.shared_encoder.embedding(input_ids)
        if segment_ids is not None:
            x = x + self.shared_encoder.segment_embedding(segment_ids)
        x = self.shared_encoder.positional_encoding(x)
        
        # Transformer encoder
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        
        x = self.shared_encoder.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.shared_encoder.norm(x)
        
        # [CLS]トークンの表現を取得
        cls_repr = x[:, 0, :]  # (batch, d_model)
        cls_repr_dropout = self.shared_encoder.dropout(cls_repr)
        
        return cls_repr_dropout
    
    def select_action(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        return_log_prob: bool = True
    ) -> Tuple[Dict[str, Any], torch.Tensor, Optional[torch.Tensor]]:
        """
        アクションを選択（推論用）
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            segment_ids: Segment IDs
            temperature: Temperature for sampling
            return_log_prob: Whether to return log probability
        
        Returns:
            (action_dict, log_prob, value)
        """
        with torch.no_grad():
            # モデル推論
            main_logits, arg1_logits, arg2_logits, value, _ = self.forward(
                input_ids, attention_mask, segment_ids, return_pretrained_logits=False
            )
            
            # 温度を適用してサンプリング
            main_probs = F.softmax(main_logits / temperature, dim=-1)
            arg1_probs = F.softmax(arg1_logits / temperature, dim=-1)
            arg2_probs = F.softmax(arg2_logits / temperature, dim=-1)
            
            # サンプリング
            main_action = torch.multinomial(main_probs, 1).squeeze(-1)
            arg1_action = torch.multinomial(arg1_probs, 1).squeeze(-1)
            arg2_action = torch.multinomial(arg2_probs, 1).squeeze(-1)
            
            # アクション辞書を作成
            action_dict = {
                'main': main_action.item(),
                'arg1': arg1_action.item(),
                'arg2': arg2_action.item()
            }
            
            # ログ確率を計算
            log_prob = None
            if return_log_prob:
                main_log_prob = F.log_softmax(main_logits / temperature, dim=-1)
                arg1_log_prob = F.log_softmax(arg1_logits / temperature, dim=-1)
                arg2_log_prob = F.log_softmax(arg2_logits / temperature, dim=-1)
                
                log_prob = (
                    main_log_prob[0, main_action].item() +
                    arg1_log_prob[0, arg1_action].item() +
                    arg2_log_prob[0, arg2_action].item()
                )
            
            return action_dict, log_prob, value.item()
    
    def compute_kl_divergence(
        self,
        actor_logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        pretrained_logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Actorとpretrained modelの間のKL divergenceを計算
        
        Args:
            actor_logits: (main_logits, arg1_logits, arg2_logits)
            pretrained_logits: (pretrained_main_logits, pretrained_arg1_logits, pretrained_arg2_logits)
        
        Returns:
            KL divergence loss
        """
        actor_main, actor_arg1, actor_arg2 = actor_logits
        pretrained_main, pretrained_arg1, pretrained_arg2 = pretrained_logits
        
        # KL divergenceを計算
        kl_main = F.kl_div(
            F.log_softmax(actor_main, dim=-1),
            F.softmax(pretrained_main, dim=-1),
            reduction='batchmean'
        )
        
        kl_arg1 = F.kl_div(
            F.log_softmax(actor_arg1, dim=-1),
            F.softmax(pretrained_arg1, dim=-1),
            reduction='batchmean'
        )
        
        kl_arg2 = F.kl_div(
            F.log_softmax(actor_arg2, dim=-1),
            F.softmax(pretrained_arg2, dim=-1),
            reduction='batchmean'
        )
        
        return kl_main + kl_arg1 + kl_arg2
    
    def compute_entropy(
        self,
        main_logits: torch.Tensor,
        arg1_logits: torch.Tensor,
        arg2_logits: torch.Tensor,
        arg1_valid_mask: torch.Tensor,
        arg2_valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        エントロピーを計算
        
        Args:
            main_logits: Main tactic logits
            arg1_logits: Arg1 logits
            arg2_logits: Arg2 logits
            arg1_valid_mask: Valid arg1 mask
            arg2_valid_mask: Valid arg2 mask
        
        Returns:
            Entropy loss
        """
        # Main tactic entropy
        main_probs = F.softmax(main_logits, dim=-1)
        main_log_probs = F.log_softmax(main_logits, dim=-1)
        main_entropy = -(main_probs * main_log_probs).sum(dim=-1).mean()
        
        # Arg1 entropy (only for valid ones)
        arg1_entropy = torch.tensor(0.0, device=main_logits.device)
        if arg1_valid_mask.any():
            arg1_probs = F.softmax(arg1_logits, dim=-1)
            arg1_log_probs = F.log_softmax(arg1_logits, dim=-1)
            arg1_entropy = -(arg1_probs[arg1_valid_mask] * arg1_log_probs[arg1_valid_mask]).sum(dim=-1).mean()
        
        # Arg2 entropy (only for valid ones)
        arg2_entropy = torch.tensor(0.0, device=main_logits.device)
        if arg2_valid_mask.any():
            arg2_probs = F.softmax(arg2_logits, dim=-1)
            arg2_log_probs = F.log_softmax(arg2_logits, dim=-1)
            arg2_entropy = -(arg2_probs[arg2_valid_mask] * arg2_log_probs[arg2_valid_mask]).sum(dim=-1).mean()
        
        return main_entropy + arg1_entropy + arg2_entropy


def create_actor_critic_model(
    pretrained_model_path: str,
    vocab_size: int,
    pad_id: int,
    max_seq_len: int,
    num_main_classes: int,
    num_arg1_classes: int,
    num_arg2_classes: int,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    critic_hidden_dim: int = 128,
    device: torch.device = None
) -> ActorCriticModel:
    """
    Actor-Criticモデルを作成するヘルパー関数
    
    Args:
        pretrained_model_path: 事前学習済みモデルのパス
        vocab_size: 語彙サイズ
        pad_id: パディングID
        max_seq_len: 最大シーケンス長
        num_main_classes: メインタクティククラス数
        num_arg1_classes: Arg1クラス数
        num_arg2_classes: Arg2クラス数
        d_model: モデル次元
        nhead: アテンションヘッド数
        num_layers: レイヤー数
        dim_feedforward: フィードフォワード次元
        dropout: ドロップアウト率
        critic_hidden_dim: Critic隠れ層次元
        device: デバイス
    
    Returns:
        ActorCriticModel
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ベースTransformerを作成
    base_transformer = TransformerClassifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_seq_len=max_seq_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_main_classes=num_main_classes,
        num_arg1_classes=num_arg1_classes,
        num_arg2_classes=num_arg2_classes,
    )
    
    # 事前学習済みモデルを読み込み
    pretrained_model = TransformerClassifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_seq_len=max_seq_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_main_classes=num_main_classes,
        num_arg1_classes=num_arg1_classes,
        num_arg2_classes=num_arg2_classes,
    )
    
    # 事前学習済み重みを読み込み
    if pretrained_model_path and torch.cuda.is_available():
        state_dict = torch.load(pretrained_model_path, map_location=device)
        pretrained_model.load_state_dict(state_dict)
    elif pretrained_model_path:
        state_dict = torch.load(pretrained_model_path, map_location='cpu')
        pretrained_model.load_state_dict(state_dict)
    
    # ベースTransformerに事前学習済み重みをコピー
    base_transformer.load_state_dict(pretrained_model.state_dict())
    
    # デバイスに移動
    base_transformer = base_transformer.to(device)
    pretrained_model = pretrained_model.to(device)
    
    # Actor-Criticモデルを作成
    actor_critic = ActorCriticModel(
        base_transformer=base_transformer,
        pretrained_model=pretrained_model,
        critic_hidden_dim=critic_hidden_dim,
        dropout=dropout
    )
    
    return actor_critic
