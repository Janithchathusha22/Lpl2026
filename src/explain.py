"""
LPL 2026 — Explainable AI (XAI)
Gradient-based Saliency feature attribution for the PyTorch MT-DNN.

Attribution formula:
  contribution_i = (∂ŷ / ∂x_i) × x_i
  (Vanilla Gradient × Input — signed, shows direction of influence)
"""

import os
import json
import numpy as np
import torch
import sys

sys.path.insert(0, os.path.dirname(__file__))
from models import LPLMatchWinnerNN

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

FEATURE_LABELS = {
    "team1_bat_rating":      "Team 1 Batting Strength",
    "team1_bowl_rating":     "Team 1 Bowling Strength",
    "team2_bat_rating":      "Team 2 Batting Strength",
    "team2_bowl_rating":     "Team 2 Bowling Strength",
    "team1_is_home":         "Team 1 Home Advantage",
    "team2_is_home":         "Team 2 Home Advantage",
    "venue_bat_modifier":    "Venue Bat Modifier",
    "venue_spin_modifier":   "Venue Spin Factor",
    "venue_pace_modifier":   "Venue Pace Factor",
    "venue_dew_factor":      "Dew Factor",
    "team1_exp_rating":      "Team 1 Experience",
    "team2_exp_rating":      "Team 2 Experience",
    "bat_rating_diff":       "Batting Gap (T1 - T2)",
    "bowl_rating_diff":      "Bowling Gap (T1 - T2)",
    "exp_rating_diff":       "Experience Gap (T1 - T2)",
    "home_advantage_diff":   "Home Advantage Gap",
}


def load_pytorch_model(models_dir=MODELS_DIR):
    """Load the saved MT-DNN and normalization parameters."""
    model_path = os.path.join(models_dir, "pytorch_mtdnn.pth")
    norm_path  = os.path.join(models_dir, "norm_params.json")

    if not os.path.exists(model_path):
        return None, None

    model = LPLMatchWinnerNN(input_dim=16, dropout_rate=0.2)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    norm_params = None
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            norm_params = json.load(f)

    return model, norm_params


def normalize_input(feature_dict, norm_params):
    """Normalize a raw feature dict using stored min-max params."""
    feat_cols = norm_params["feature_cols"]
    feat_min  = np.array(norm_params["feat_min"], dtype=np.float32)
    feat_max  = np.array(norm_params["feat_max"], dtype=np.float32)
    denom     = feat_max - feat_min
    denom[denom == 0] = 1.0

    raw = np.array([float(feature_dict.get(c, 0.0)) for c in feat_cols], dtype=np.float32)
    norm = (raw - feat_min) / denom
    return torch.tensor(norm, dtype=torch.float32).unsqueeze(0)


def predict_with_xai(feature_dict, model=None, norm_params=None, models_dir=MODELS_DIR):
    """
    Run inference on the MT-DNN and compute Gradient × Input attributions.

    Returns:
        win_prob      (float)  — Team 1 win probability 0–100
        score_pred    (list)   — [team1_score, team2_score]
        wickets_pred  (list)   — [team1_wickets, team2_wickets]
        attributions  (dict)   — {feature_label: contribution_value}
    """
    if model is None:
        model, norm_params = load_pytorch_model(models_dir)
    if model is None:
        return None, None, None, None

    feat_cols = norm_params["feature_cols"]
    x_tensor = normalize_input(feature_dict, norm_params)
    x_tensor.requires_grad_(True)

    model.eval()
    win_prob_raw, scores_raw, wickets_raw = model(x_tensor)

    # Backprop on win probability head (scalar output)
    win_prob_raw.backward()

    # Gradient × Input attribution
    grads       = x_tensor.grad.detach().numpy()[0]
    inputs      = x_tensor.detach().numpy()[0]
    attribution = grads * inputs   # Grad × Input: direction + magnitude

    feat_min = np.array(norm_params["feat_min"], dtype=np.float32)
    feat_max = np.array(norm_params["feat_max"], dtype=np.float32)

    attribution_dict = {}
    for i, col in enumerate(feat_cols):
        label = FEATURE_LABELS.get(col, col)
        attribution_dict[label] = float(attribution[i])

    win_prob   = float(win_prob_raw.detach().item()) * 100.0
    score_pred = [
        float(scores_raw[0][0].detach()) * 200.0,
        float(scores_raw[0][1].detach()) * 200.0,
    ]
    wkts_pred  = [
        float(wickets_raw[0][0].detach()) * 10.0,
        float(wickets_raw[0][1].detach()) * 10.0,
    ]

    return win_prob, score_pred, wkts_pred, attribution_dict


def predict_player_performance(player_row_dict, venue_name, models_dir=MODELS_DIR):
    """
    Run the PP-NN for a single player at a specific venue.
    Returns {expected_runs, expected_sr, expected_wickets, expected_economy}.
    """
    from data_prep import get_player_features_for_venue, VENUE_MAP
    from models import LPLPlayerPerformanceNN

    ppnn_path  = os.path.join(models_dir, "pytorch_ppnn.pth")
    norm_path  = os.path.join(models_dir, "ppnn_norm_params.json")

    if not os.path.exists(ppnn_path):
        return None

    ppnn = LPLPlayerPerformanceNN(input_dim=10, dropout_rate=0.2)
    ppnn.load_state_dict(torch.load(ppnn_path, map_location="cpu", weights_only=True))
    ppnn.eval()

    with open(norm_path) as f:
        norm = json.load(f)

    x = get_player_features_for_venue(player_row_dict, venue_name).unsqueeze(0)
    x_min = torch.tensor(norm["x_min"])
    x_max = torch.tensor(norm["x_max"])
    denom = x_max - x_min
    denom[denom == 0] = 1.0
    x_norm = (x - x_min) / denom

    with torch.no_grad():
        bat_out  = ppnn(x_norm, is_batsman=True)[0]
        bowl_out = ppnn(x_norm, is_batsman=False)[0]

    bs, ss = norm["bat_scale"]
    ws, es = norm["bowl_scale"]

    return {
        "expected_runs":     round(float(bat_out[0]) * bs, 1),
        "expected_sr":       round(float(bat_out[1]) * ss, 1),
        "expected_wickets":  round(float(bowl_out[0]) * ws, 2),
        "expected_economy":  round(float(bowl_out[1]) * es, 2),
    }


if __name__ == "__main__":
    feat = {
        "team1_bat_rating": 84, "team1_bowl_rating": 88,
        "team2_bat_rating": 75, "team2_bowl_rating": 90,
        "team1_is_home": 1, "team2_is_home": 0,
        "venue_bat_modifier": 1.03, "venue_spin_modifier": 1.05, "venue_pace_modifier": 1.08,
        "venue_dew_factor": 0, "team1_exp_rating": 88, "team2_exp_rating": 79,
        "bat_rating_diff": 9, "bowl_rating_diff": -2, "exp_rating_diff": 9, "home_advantage_diff": 1
    }
    win_p, scores, wkts, attribs = predict_with_xai(feat)
    if win_p is not None:
        print(f"Win Probability Team 1: {win_p:.1f}%")
        print(f"Projected Scores: {scores[0]:.0f} vs {scores[1]:.0f}")
        print(f"Projected Wickets: {wkts[0]:.1f} vs {wkts[1]:.1f}")
        print("\nXAI - Top Feature Attributions:")
        for k, v in sorted(attribs.items(), key=lambda x: abs(x[1]), reverse=True)[:8]:
            bar = "+" if v >= 0 else "-"
            print(f"  {bar} {k:<30}: {v:+.4f}")
