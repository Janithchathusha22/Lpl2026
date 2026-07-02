"""
LPL 2026 Prediction System — PyTorch Neural Network Training
Trains LPLMatchWinnerNN for 60 epochs with:
- Multi-task loss: BCE (winner) + MSE (scores) + MSE (wickets)
- Adam optimizer, BatchNorm, Dropout
- Saves model weights + normalization params to models/
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, PoissonRegressor
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(__file__))
from models import LPLMatchWinnerNN, LPLPlayerPerformanceNN
from data_prep import MatchDataset, build_player_master

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class LPLModelTrainer:
    def __init__(self, data_dir="models"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. PREPARE MATCH TRAINING MATRIX
    # ─────────────────────────────────────────────────────────────────────────
    def prepare_datasets(self, df):
        WINNER_FEATS = [
            'team1_bat_rating', 'team1_bowl_rating',
            'team2_bat_rating', 'team2_bowl_rating',
            'team1_is_home', 'team2_is_home',
            'venue_bat_modifier', 'venue_spin_modifier', 'venue_pace_modifier',
            'venue_dew_factor', 'team1_exp_rating', 'team2_exp_rating',
            'bat_rating_diff', 'bowl_rating_diff', 'exp_rating_diff', 'home_advantage_diff'
        ]
        for c in WINNER_FEATS:
            if c not in df.columns:
                df[c] = 0.0

        X_win = df[WINNER_FEATS]
        y_win = df['team1_won'].astype(int)
        X_score  = df[['team1_bat_rating','team2_bowl_rating','venue_bat_modifier',
                        'venue_dew_factor','team1_exp_rating','team2_exp_rating']]
        y_score  = df['team1_score'].astype(float)
        X_wkts   = df[['team1_bat_rating','team2_bowl_rating','venue_spin_modifier',
                        'venue_pace_modifier','venue_dew_factor']]
        y_wkts   = df['team1_wickets'].astype(float)
        return (X_win, y_win), (X_score, y_score), (X_wkts, y_wkts)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. TRAIN CLASSICAL sklearn MODELS (baseline)
    # ─────────────────────────────────────────────────────────────────────────
    def train_classical(self, df):
        print("\n> [Classical] Training baseline sklearn models...")
        (X_win, y_win), (X_score, y_score), (X_wkts, y_wkts) = self.prepare_datasets(df)

        Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(X_win, y_win, test_size=0.2, random_state=42)
        Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X_score, y_score, test_size=0.2, random_state=42)
        Xk_tr, Xk_te, yk_tr, yk_te = train_test_split(X_wkts, y_wkts, test_size=0.2, random_state=42)

        lr_winner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr_winner.fit(Xw_tr, yw_tr)

        rf_score = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf_score.fit(Xs_tr, ys_tr)

        poisson_wkts = PoissonRegressor(alpha=1.0)
        poisson_wkts.fit(Xk_tr, yk_tr)

        for name, obj in [
            ("lr_winner.pkl",        lr_winner),
            ("rf_score.pkl",         rf_score),
            ("poisson_wickets.pkl",  poisson_wkts),
            ("feature_names_win.pkl",   list(X_win.columns)),
            ("feature_names_score.pkl", list(X_score.columns)),
            ("feature_names_wickets.pkl", list(X_wkts.columns)),
        ]:
            with open(os.path.join(self.data_dir, name), "wb") as f:
                pickle.dump(obj, f)

        print("  [ok] Classical models saved.")
        return Xw_te, yw_te, Xs_te, ys_te, Xk_te, yk_te

    # ─────────────────────────────────────────────────────────────────────────
    # 3. TRAIN PyTorch MULTI-TASK DEEP NEURAL NETWORK (60 EPOCHS)
    # ─────────────────────────────────────────────────────────────────────────
    def train_pytorch_mtdnn(self, df, epochs=60, batch_size=16, lr=0.003):
        print(f"\n> [PyTorch MT-DNN] Training for {epochs} epochs...\n")
        print("  Architecture: Input(16) -> Shared[128->64] -> 3 Heads [Winner|Score|Wickets]")
        print("  Optimizer: Adam (lr=0.003)")
        print("  Loss: 0.6*BCE + 0.3*MSE(scores) + 0.1*MSE(wickets)\n")
        print(f"  {'Epoch':>5} | {'Train Loss':>12} | {'Win BCE':>10} | {'Score MSE':>10} | {'Wkts MSE':>10}")
        print("  " + "-" * 60)

        dataset = MatchDataset(df, fit=True)

        # Store normalization params for inference
        norm_params = {
            "feat_min": dataset.feat_min.tolist(),
            "feat_max": dataset.feat_max.tolist(),
            "feature_cols": dataset.FEATURE_COLS,
        }

        # Normalize features
        Xn = dataset.normalize(dataset.X, dataset.feat_min, dataset.feat_max)
        dataset.X = Xn  # Replace raw with normalized

        n_train = int(0.8 * len(dataset))
        n_val   = len(dataset) - n_train
        train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                        generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size)

        model = LPLMatchWinnerNN(input_dim=16, dropout_rate=0.2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

        # Score normalization constants (divide by max expected score to keep MSE reasonable)
        SCORE_SCALE  = 200.0
        WICKETS_SCALE = 10.0

        best_val_loss = float("inf")
        history = []

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = epoch_bce = epoch_mse_s = epoch_mse_k = 0.0

            for X_b, y_w, y_s, y_k in train_loader:
                optimizer.zero_grad()

                pred_w, pred_s, pred_k = model(X_b)

                loss_w = bce_loss(pred_w, y_w)
                loss_s = mse_loss(pred_s, y_s / SCORE_SCALE)
                loss_k = mse_loss(pred_k, y_k / WICKETS_SCALE)

                loss = 0.6 * loss_w + 0.3 * loss_s + 0.1 * loss_k
                loss.backward()
                optimizer.step()

                epoch_loss   += loss.item()
                epoch_bce    += loss_w.item()
                epoch_mse_s  += loss_s.item()
                epoch_mse_k  += loss_k.item()

            scheduler.step()

            n_batches = len(train_loader)
            avg_loss = epoch_loss / n_batches
            history.append({"epoch": epoch, "train_loss": avg_loss})

            print(f"  {epoch:>5} | {avg_loss:>12.6f} | "
                  f"{epoch_bce/n_batches:>10.6f} | "
                  f"{epoch_mse_s/n_batches:>10.6f} | "
                  f"{epoch_mse_k/n_batches:>10.6f}")

        # Save model
        torch.save(model.state_dict(), os.path.join(self.data_dir, "pytorch_mtdnn.pth"))
        with open(os.path.join(self.data_dir, "norm_params.json"), "w") as f:
            json.dump(norm_params, f, indent=2)
        with open(os.path.join(self.data_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        print(f"\n  [ok] PyTorch MT-DNN saved -> models/pytorch_mtdnn.pth")
        print(f"  [ok] Normalization params -> models/norm_params.json")
        return model, norm_params, val_loader, SCORE_SCALE, WICKETS_SCALE

    # ─────────────────────────────────────────────────────────────────────────
    # 4. TRAIN PLAYER PERFORMANCE NN
    # ─────────────────────────────────────────────────────────────────────────
    def train_player_ppnn(self, player_master_df, epochs=60, lr=0.003):
        """
        Trains the per-player performance neural network using batting + bowling profiles.
        Uses venue suitability + phase skill ratings as input features.
        """
        print(f"\n> [PP-NN] Training Player Performance NN for {epochs} epochs...")

        df = player_master_df.dropna(subset=["batting_power_play_rating"])

        # Feature columns for PP-NN
        pp_feat_cols = [
            "batting_power_play_rating", "batting_middle_over_rating", "batting_death_over_rating",
            "bowling_power_play_rating", "bowling_middle_over_rating", "bowling_death_over_rating",
            "premadasa_rating", "spin_handling", "pace_handling", "pressure_handling"
        ]
        for c in pp_feat_cols:
            if c not in df.columns:
                df[c] = 5.0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(5.0)

        # Batting targets: use powerplay_sr + avg as proxy runs & sr
        if "powerplay_sr_estimate" not in df.columns:
            df["powerplay_sr_estimate"] = 130.0
        if "avg_estimate" not in df.columns:
            df["avg_estimate"] = 20.0
        if "wickets_per_match_estimate" not in df.columns:
            df["wickets_per_match_estimate"] = 0.0
        if "economy_middle" not in df.columns:
            df["economy_middle"] = 8.0

        X = torch.tensor(df[pp_feat_cols].values, dtype=torch.float32)
        # Normalize
        x_min = X.min(dim=0).values
        x_max = X.max(dim=0).values
        denom = x_max - x_min
        denom[denom == 0] = 1
        X_norm = (X - x_min) / denom

        y_bat = torch.tensor(
            df[["powerplay_sr_estimate", "avg_estimate"]].values / [200.0, 50.0],
            dtype=torch.float32
        )
        y_bowl = torch.tensor(
            df[["wickets_per_match_estimate", "economy_middle"]].values / [5.0, 12.0],
            dtype=torch.float32
        )

        model = LPLPlayerPerformanceNN(input_dim=10, dropout_rate=0.2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        mse = nn.MSELoss()

        print(f"  {'Epoch':>5} | {'Bat Loss':>10} | {'Bowl Loss':>10}")
        print("  " + "-" * 35)

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()

            pred_bat  = model(X_norm, is_batsman=True)
            pred_bowl = model(X_norm, is_batsman=False)

            loss_bat  = mse(pred_bat,  y_bat)
            loss_bowl = mse(pred_bowl, y_bowl)
            loss = 0.5 * loss_bat + 0.5 * loss_bowl
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0 or epoch == 1:
                print(f"  {epoch:>5} | {loss_bat.item():>10.6f} | {loss_bowl.item():>10.6f}")

        # Save PP-NN
        torch.save(model.state_dict(), os.path.join(self.data_dir, "pytorch_ppnn.pth"))
        ppnn_norm = {
            "x_min": x_min.tolist(), "x_max": x_max.tolist(),
            "feature_cols": pp_feat_cols,
            "bat_scale": [200.0, 50.0],
            "bowl_scale": [5.0, 12.0]
        }
        import json
        with open(os.path.join(self.data_dir, "ppnn_norm_params.json"), "w") as f:
            json.dump(ppnn_norm, f, indent=2)
        print(f"  [ok] PP-NN saved -> models/pytorch_ppnn.pth")
        return model

    def train_and_save(self, df):
        """Compatibility shim: trains classical models only."""
        classical_results = self.train_classical(df)
        return classical_results


# Needed for direct import of pandas in train_player_ppnn
try:
    import pandas as pd
except ImportError:
    pass


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from loader import LPLDataLoader
    from preprocess import LPLFeatureEngineer
    from data_prep import build_player_master

    loader = LPLDataLoader()
    datasets = loader.load_all_data()

    fe = LPLFeatureEngineer(datasets)
    processed_df = fe.prepare_match_features()

    trainer = LPLModelTrainer(data_dir=MODELS_DIR)
    trainer.train_classical(processed_df)
    trainer.train_pytorch_mtdnn(processed_df, epochs=60)

    player_master = build_player_master(datasets)
    if player_master is not None:
        trainer.train_player_ppnn(player_master, epochs=60)
