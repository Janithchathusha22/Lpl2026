"""
LPL 2026 Prediction System — run_pipeline.py
Single entry-point to:
  1. Load all 21 CSV files
  2. Build unified player master (all files connected)
  3. Train classical sklearn baseline models
  4. Train PyTorch MT-DNN for 60 epochs
  5. Train Player Performance NN for 60 epochs
  6. Evaluate all models and print metrics
"""

import os
import sys
import json
import pickle
import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from loader import LPLDataLoader
from preprocess import LPLFeatureEngineer
from data_prep import build_player_master
from train import LPLModelTrainer, MODELS_DIR

# Evaluation helpers
from sklearn.metrics import (accuracy_score, log_loss, mean_squared_error,
                              mean_absolute_error)
from sklearn.model_selection import train_test_split
import torch
from models import LPLMatchWinnerNN
from data_prep import MatchDataset


def evaluate_models(datasets, processed_df, player_master):
    print("\n" + "="*60)
    print("       LPL 2026 - Model Evaluation Results")
    print("="*60)

    # ── Classical sklearn ───────────────────────────────────────────────────
    def load_pkl(name):
        p = os.path.join(MODELS_DIR, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
        return None

    feat_win   = load_pkl("feature_names_win.pkl")
    lr_winner  = load_pkl("lr_winner.pkl")
    rf_score   = load_pkl("rf_score.pkl")
    poisson_wk = load_pkl("poisson_wickets.pkl")

    if feat_win and lr_winner:
        X_win = processed_df[feat_win]
        y_win = processed_df['team1_won'].astype(int)
        _, X_te, _, y_te = train_test_split(X_win, y_win, test_size=0.2, random_state=42)
        y_pred = lr_winner.predict(X_te)
        y_prob = lr_winner.predict_proba(X_te)[:, 1]
        print(f"\n  Logistic Regression (Baseline):")
        print(f"    Accuracy  : {accuracy_score(y_te, y_pred)*100:.2f}%")
        print(f"    Log-Loss  : {log_loss(y_te, y_prob):.4f}")

    feat_s = load_pkl("feature_names_score.pkl")
    if feat_s and rf_score:
        X_sc = processed_df[feat_s]
        y_sc = processed_df['team1_score'].astype(float)
        _, X_te, _, y_te = train_test_split(X_sc, y_sc, test_size=0.2, random_state=42)
        y_pred = rf_score.predict(X_te)
        print(f"\n  Random Forest Regressor (Score):")
        print(f"    RMSE : {np.sqrt(mean_squared_error(y_te, y_pred)):.2f} runs")
        print(f"    MAE  : {mean_absolute_error(y_te, y_pred):.2f} runs")

    # ── PyTorch MT-DNN ──────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "pytorch_mtdnn.pth")
    norm_path  = os.path.join(MODELS_DIR, "norm_params.json")

    if os.path.exists(model_path) and os.path.exists(norm_path):
        model = LPLMatchWinnerNN(input_dim=16, dropout_rate=0.2)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()

        with open(norm_path) as f:
            norm = json.load(f)

        dataset = MatchDataset(processed_df, fit=True)
        Xn = dataset.normalize(dataset.X,
                               np.array(norm["feat_min"], dtype=np.float32),
                               np.array(norm["feat_max"], dtype=np.float32))
        dataset.X = Xn

        from torch.utils.data import random_split, DataLoader
        n_train = int(0.8 * len(dataset))
        _, val_ds = random_split(dataset, [n_train, len(dataset)-n_train],
                                 generator=torch.Generator().manual_seed(42))
        val_loader = DataLoader(val_ds, batch_size=32)

        all_win_true, all_win_pred, all_win_prob = [], [], []
        all_s_true, all_s_pred = [], []
        all_k_true, all_k_pred = [], []

        with torch.no_grad():
            for X_b, y_w, y_s, y_k in val_loader:
                pw, ps, pk = model(X_b)
                prob = pw.numpy().flatten()
                pred = (prob >= 0.5).astype(int)
                all_win_true.extend(y_w.numpy().flatten().tolist())
                all_win_prob.extend(prob.tolist())
                all_win_pred.extend(pred.tolist())
                all_s_true.extend(y_s.numpy().tolist())
                all_s_pred.extend((ps.numpy() * 200.0).tolist())
                all_k_true.extend(y_k.numpy().tolist())
                all_k_pred.extend((pk.numpy() * 10.0).tolist())

        acc  = accuracy_score(all_win_true, all_win_pred)
        ll   = log_loss(all_win_true, all_win_prob)
        s_tr = np.array(all_s_true)
        s_pr = np.array(all_s_pred)
        k_tr = np.array(all_k_true)
        k_pr = np.array(all_k_pred)

        print(f"\n  PyTorch MT-DNN (Winner Head):")
        print(f"    Accuracy  : {acc*100:.2f}%")
        print(f"    Log-Loss  : {ll:.4f}")
        print(f"\n  PyTorch MT-DNN (Score Head):")
        print(f"    RMSE T1   : {np.sqrt(mean_squared_error(s_tr[:,0], s_pr[:,0])):.2f} runs")
        print(f"    MAE  T1   : {mean_absolute_error(s_tr[:,0], s_pr[:,0]):.2f} runs")
        print(f"\n  PyTorch MT-DNN (Wickets Head):")
        print(f"    RMSE T1   : {np.sqrt(mean_squared_error(k_tr[:,0], k_pr[:,0])):.2f} wickets")
        print(f"    MAE  T1   : {mean_absolute_error(k_tr[:,0], k_pr[:,0]):.2f} wickets")

    print("\n" + "="*60)
    print("  [ok] All Models Evaluated Successfully!")
    print("="*60 + "\n")


def main():
    print("\n" + "="*60)
    print("      LPL 2026 ML Pipeline - Starting")
    print("="*60)

    # ── Step 1: Load all 21 datasets ────────────────────────────────────────
    print("\n[1/5] Loading all CSV datasets...")
    loader   = LPLDataLoader()
    datasets = loader.load_all_data()
    print(f"  Loaded {sum(1 for v in datasets.values() if v is not None)} datasets successfully.")

    # ── Step 2: Feature Engineering ─────────────────────────────────────────
    print("\n[2/5] Feature engineering & preprocessing...")
    fe           = LPLFeatureEngineer(datasets)
    processed_df = fe.prepare_match_features()
    print(f"  Match simulation matrix: {processed_df.shape[0]} rows x {processed_df.shape[1]} cols")

    # ── Step 3: Build Unified Player Master ──────────────────────────────────
    print("\n[3/5] Building unified player master (all files connected)...")
    player_master = build_player_master(datasets)
    if player_master is not None:
        print(f"  Player master: {player_master.shape[0]} players x {player_master.shape[1]} features")
    else:
        print("  [warn] Player master could not be built - some datasets missing.")

    # ── Step 4: Train Models ─────────────────────────────────────────────────
    print("\n[4/5] Training all models...")
    trainer = LPLModelTrainer(data_dir=MODELS_DIR)
    trainer.train_classical(processed_df)
    trainer.train_pytorch_mtdnn(processed_df, epochs=60, batch_size=16, lr=0.003)
    if player_master is not None:
        trainer.train_player_ppnn(player_master, epochs=60, lr=0.003)

    # ── Step 5: Evaluate ─────────────────────────────────────────────────────
    print("\n[5/5] Running evaluation...")
    evaluate_models(datasets, processed_df, player_master)


if __name__ == "__main__":
    main()
