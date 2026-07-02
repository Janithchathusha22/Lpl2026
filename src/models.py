"""
LPL 2026 Prediction System - PyTorch Multi-Task Deep Neural Network
Handles: Match Winner (Classification) + Score Prediction (Regression) + Wickets (Regression)
All connected through shared representation learning across 16 engineered features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LPLMatchWinnerNN(nn.Module):
    """
    Multi-Task Deep Neural Network for LPL 2026 match predictions.
    
    Architecture:
        Input (16 features) 
        → Shared Encoder [Dense(16→128) → BatchNorm → ReLU → Dropout(0.2)]
        → Shared Layer 2 [Dense(128→64) → BatchNorm → ReLU]
        → Three parallel output heads:
            1. Winner Head   → Sigmoid → Win Probability %
            2. Score Head    → Linear  → [team1_score, team2_score]
            3. Wickets Head  → Softplus → [team1_wickets, team2_wickets]
    """
    def __init__(self, input_dim=16, dropout_rate=0.2):
        super(LPLMatchWinnerNN, self).__init__()
        
        # === SHARED ENCODER ===
        self.shared_layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.shared_layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        # === TASK-SPECIFIC OUTPUT HEADS ===
        
        # 1. Winner Head: Binary Classification → Win Probability 0-1
        self.winner_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 2. Score Prediction Head: Regression → [team1_score, team2_score]
        self.score_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)   # Linear output for regression
        )
        
        # 3. Wickets Prediction Head: Regression → [team1_wickets, team2_wickets]
        self.wickets_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softplus()       # Softplus ensures positive outputs (counts)
        )
        
    def forward(self, x):
        # Shared encoding
        x = self.shared_layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.shared_layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Multi-task outputs
        win_prob = self.winner_head(x)          # shape: (batch, 1)
        scores = self.score_head(x)             # shape: (batch, 2) → [t1_score, t2_score]
        wickets = self.wickets_head(x)          # shape: (batch, 2) → [t1_wkts, t2_wkts]
        
        return win_prob, scores, wickets


class LPLPlayerPerformanceNN(nn.Module):
    """
    Player-specific Performance Neural Network (PP-NN).
    
    Predicts per-player expected output at a specific venue:
        - For batters: Expected Runs, Expected Strike Rate
        - For bowlers: Expected Wickets, Expected Economy Rate
    
    Input (10 features):
        Phase ratings (PP/Middle/Death) × 2 (bat+bowl)
        Venue suitability rating (1 of 4 venues)
        spin_handling, pace_handling, pressure_avg
        bowling_type encoding (1 hot: 3 values)
    """
    def __init__(self, input_dim=10, dropout_rate=0.2):
        super(LPLPlayerPerformanceNN, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Batting output head
        self.bat_output = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),       # [expected_runs, expected_sr]
            nn.Softplus()           # Positive outputs
        )
        
        # Bowling output head
        self.bowl_output = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),       # [expected_wickets, expected_economy]
            nn.Softplus()           # Positive outputs
        )

    def forward(self, x, is_batsman=True):
        z = self.encoder(x)
        if is_batsman:
            return self.bat_output(z)     # → [runs, strike_rate]
        else:
            return self.bowl_output(z)    # → [wickets, economy]
