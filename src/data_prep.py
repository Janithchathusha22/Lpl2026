"""
LPL 2026 Prediction System — Unified Data Preparation
Merges all 21 CSV files by player_id/team/venue keys
into a unified player profile frame + match training matrix.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


VENUE_MAP = {
    "Premadasa": "premadasa_rating",
    "R. Premadasa Stadium": "premadasa_rating",
    "R.Premadasa Stadium": "premadasa_rating",
    "SSC Ground": "premadasa_rating",
    "Galle": "galle_rating",
    "Galle International Stadium": "galle_rating",
    "Pallekele": "pallekele_rating",
    "Pallekele Stadium": "pallekele_rating",
    "Pallekele International Stadium": "pallekele_rating",
    "Dambulla": "dambulla_rating",
    "Rangiri Dambulla": "dambulla_rating",
    "Rangiri Dambulla International": "dambulla_rating",
}


def build_player_master(datasets):
    """
    Joins LPL_2026_Full_Squad → 01_player_master → 02_player_skill_ratings
    → 05_player_venue_suitability → 06_bowling_matchup_matrix
    → 07_batting_profiles → 08_category_breakdown
    Returns: Unified player profile dataframe (one row per player).
    """
    squad  = datasets.get("full_squad")
    master = datasets.get("player_master")
    skills = datasets.get("player_skill_ratings")
    venue  = datasets.get("player_venue_suitability")
    bowl   = datasets.get("bowling_matchup_matrix")
    bat    = datasets.get("batting_profiles")
    cat_bk = datasets.get("category_breakdown")

    if master is None or skills is None:
        return None

    # 1. Merge player_master + skills on player_id
    df = pd.merge(master, skills, on=["player_id", "player_name", "team", "category"], how="left")

    # 2. Merge venue suitability (player_id present)
    if venue is not None:
        venue_cols = ["player_id", "premadasa_rating", "galle_rating",
                      "pallekele_rating", "dambulla_rating", "best_venue", "worst_venue"]
        df = pd.merge(df, venue[venue_cols], on="player_id", how="left")
    else:
        for c in ["premadasa_rating","galle_rating","pallekele_rating","dambulla_rating"]:
            df[c] = 7.0  # neutral default

    # 3. Merge bowling matchup matrix (bowler_id == player_id)
    if bowl is not None:
        bowl_rename = bowl.rename(columns={"bowler_id": "player_id", "bowler_name": "player_name"})
        bowl_cols = ["player_id", "vs_RHB_powerplay", "vs_LHB_powerplay",
                     "vs_RHB_middle", "vs_LHB_middle", "vs_RHB_death", "vs_LHB_death",
                     "economy_powerplay", "economy_middle", "economy_death",
                     "wickets_per_match_estimate", "best_phase", "weakness_phase"]
        df = pd.merge(df, bowl_rename[bowl_cols], on="player_id", how="left")
    for c in ["vs_RHB_powerplay","vs_LHB_powerplay","vs_RHB_middle","vs_LHB_middle",
              "vs_RHB_death","vs_LHB_death","economy_powerplay","economy_middle",
              "economy_death","wickets_per_match_estimate"]:
        if c not in df.columns:
            df[c] = 0.0
    df[["vs_RHB_powerplay","vs_LHB_powerplay","vs_RHB_middle","vs_LHB_middle",
        "vs_RHB_death","vs_LHB_death","economy_powerplay","economy_middle",
        "economy_death","wickets_per_match_estimate"]] = \
        df[["vs_RHB_powerplay","vs_LHB_powerplay","vs_RHB_middle","vs_LHB_middle",
            "vs_RHB_death","vs_LHB_death","economy_powerplay","economy_middle",
            "economy_death","wickets_per_match_estimate"]].fillna(0.0)

    # 4. Merge batting profiles
    if bat is not None:
        bat_cols = ["player_id", "batting_position", "batting_style",
                    "powerplay_sr_estimate", "middle_over_sr_estimate",
                    "death_over_sr_estimate", "avg_estimate",
                    "boundary_pct", "dot_ball_pct", "spin_handling",
                    "pace_handling", "pressure_avg", "best_match_type", "weakness"]
        df = pd.merge(df, bat[bat_cols], on="player_id", how="left")
    if "batting_style_x" in df.columns:
        df["batting_style"] = df["batting_style_x"]
    if "batting_style_y" in df.columns:
        df["batting_profile_style"] = df["batting_style_y"]
    for c in ["powerplay_sr_estimate","middle_over_sr_estimate","death_over_sr_estimate",
              "avg_estimate","boundary_pct","dot_ball_pct","spin_handling",
              "pace_handling","pressure_avg"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # 5. Merge category → impact_weight
    if cat_bk is not None and "category" in df.columns:
        cat_bk_clean = cat_bk[["category","impact_weight"]].copy()
        cat_bk_clean.columns = ["category","impact_weight"]
        df = pd.merge(df, cat_bk_clean, on="category", how="left")
    if "impact_weight" not in df.columns:
        df["impact_weight"] = 6.0
    df["impact_weight"] = pd.to_numeric(df["impact_weight"], errors="coerce").fillna(6.0)

    # 6. Numeric encode experience + t20_specialist
    df["exp_numeric"] = df["intl_experience"].map({"High": 9.0, "Medium": 7.0, "Low": 5.0}).fillna(6.0)
    df["t20_specialist_flag"] = (df["t20_specialist"].str.lower() == "yes").astype(float)

    # 7. Fill remaining numeric NaN
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    return df


def get_player_features_for_venue(player_row, venue_name):
    """
    Extracts the 10-feature vector for PP-NN given a player row and venue.
    Returns tensor of shape (10,).
    """
    venue_col = VENUE_MAP.get(venue_name, "premadasa_rating")
    feat = [
        float(player_row.get("batting_power_play_rating", 0)),
        float(player_row.get("batting_middle_over_rating", 0)),
        float(player_row.get("batting_death_over_rating", 0)),
        float(player_row.get("bowling_power_play_rating", 0)),
        float(player_row.get("bowling_middle_over_rating", 0)),
        float(player_row.get("bowling_death_over_rating", 0)),
        float(player_row.get(venue_col, 7.0)),
        float(player_row.get("spin_handling", 5.0)),
        float(player_row.get("pace_handling", 5.0)),
        float(player_row.get("pressure_avg", 5.0)),
    ]
    return torch.tensor(feat, dtype=torch.float32)


class MatchDataset(Dataset):
    """
    PyTorch Dataset wrapping the match simulation DataFrame.
    Applies min-max normalisation to all 16 feature columns.
    """
    FEATURE_COLS = [
        'team1_bat_rating', 'team1_bowl_rating',
        'team2_bat_rating', 'team2_bowl_rating',
        'team1_is_home', 'team2_is_home',
        'venue_bat_modifier', 'venue_spin_modifier', 'venue_pace_modifier',
        'venue_dew_factor',
        'team1_exp_rating', 'team2_exp_rating',
        'bat_rating_diff', 'bowl_rating_diff',
        'exp_rating_diff', 'home_advantage_diff'
    ]

    def __init__(self, df, fit=True):
        df = df.copy()
        # Derived difference features
        df['bat_rating_diff']     = df['team1_bat_rating']  - df['team2_bat_rating']
        df['bowl_rating_diff']    = df['team1_bowl_rating'] - df['team2_bowl_rating']
        df['exp_rating_diff']     = df['team1_exp_rating']  - df['team2_exp_rating']
        df['home_advantage_diff'] = df['team1_is_home'].astype(int) - df['team2_is_home'].astype(int)

        self.X = df[self.FEATURE_COLS].values.astype(np.float32)
        self.y_win     = df['team1_won'].values.astype(np.float32).reshape(-1, 1)
        self.y_scores  = df[['team1_score', 'team2_score']].values.astype(np.float32)
        self.y_wickets = df[['team1_wickets', 'team2_wickets']].values.astype(np.float32)

        # Normalize features
        if fit:
            self.feat_min  = self.X.min(axis=0)
            self.feat_max  = self.X.max(axis=0)
        # Store for external access (used during inference)
        self._fit = fit

    def normalize(self, X, feat_min, feat_max):
        denom = feat_max - feat_min
        denom[denom == 0] = 1
        return (X - feat_min) / denom

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x     = torch.tensor(self.X[idx], dtype=torch.float32)
        y_w   = torch.tensor(self.y_win[idx], dtype=torch.float32)
        y_s   = torch.tensor(self.y_scores[idx], dtype=torch.float32)
        y_k   = torch.tensor(self.y_wickets[idx], dtype=torch.float32)
        return x, y_w, y_s, y_k
