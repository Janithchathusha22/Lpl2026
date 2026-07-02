"""
LPL 2026 — FastAPI Backend Server
Serves all ML predictions via REST JSON endpoints.
Run with: uvicorn backend:app --host 127.0.0.1 --port 8000 --reload
"""

import os
import sys
import json
import math
import pickle
import random
import subprocess
import threading
from datetime import datetime
import numpy as np
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from loader import LPLDataLoader
from preprocess import (LPLFeatureEngineer, calculate_matchup_advantage,
                         calculate_win_probability, calculate_pressure_index)
from data_prep import build_player_master
from explain import predict_with_xai, predict_player_performance, load_pytorch_model
from tactical_engine import predict_tactical_plan, train_tactical_engine
import forecast_engine

MODELS_DIR = os.path.join(ROOT, "models")
FRONTEND_DIR = os.path.join(ROOT, "frontend")

# ─── Startup: Load all datasets and models ────────────────────────────────────
print("[startup] Loading datasets and models...")
_loader   = LPLDataLoader()
_datasets = _loader.load_all_data()
_fe       = LPLFeatureEngineer(_datasets)
_squad_ratings = _fe.get_team_squad_ratings()
_player_master = build_player_master(_datasets)
_pytorch_model, _norm_params = load_pytorch_model(MODELS_DIR)

def _load_pkl(name):
    p = os.path.join(MODELS_DIR, name)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

_lr_winner  = _load_pkl("lr_winner.pkl")
_feat_win   = _load_pkl("feature_names_win.pkl")
_rf_score   = _load_pkl("rf_score.pkl")
_feat_score = _load_pkl("feature_names_score.pkl")
_tactical_engine = _load_pkl("tactical_engine.pkl")

print("[startup] Backend ready.")

TEAM_VISUALS = {
    "Colombo Kaps": {"color": "#1a6ec7", "accent": "#4a9ae8", "emoji": "Lions"},
    "Galle Gallants": {"color": "#1a8a4a", "accent": "#3dcf7a", "emoji": "Swords"},
    "Jaffna Kings": {"color": "#c7a51a", "accent": "#d4bc60", "emoji": "Crown"},
    "Dambulla Sixers": {"color": "#2a4db5", "accent": "#7a9de8", "emoji": "Sixers"},
    "Kandy Royals": {"color": "#c74a1a", "accent": "#e89070", "emoji": "Royals"},
}

SUPPORTED_VENUE_ORDER = [
    "R. Premadasa Stadium",
    "Pallekele Stadium",
    "Rangiri Dambulla",
    "SSC Ground",
]

VENUE_METADATA = {
    "R. Premadasa Stadium": {"label": "Premadasa, Colombo", "city": "Colombo", "rating_col": "premadasa_rating"},
    "Pallekele Stadium": {"label": "Pallekele, Kandy", "city": "Kandy", "rating_col": "pallekele_rating"},
    "Rangiri Dambulla": {"label": "Rangiri Dambulla", "city": "Dambulla", "rating_col": "dambulla_rating"},
    "SSC Ground": {"label": "SSC Ground, Colombo", "city": "Colombo", "rating_col": "premadasa_rating"},
}

VENUE_ALIASES = {
    "Premadasa": "R. Premadasa Stadium",
    "Premadasa (Colombo)": "R. Premadasa Stadium",
    "R.Premadasa Stadium": "R. Premadasa Stadium",
    "R. Premadasa Stadium": "R. Premadasa Stadium",
    "Pallekele": "Pallekele Stadium",
    "Pallekele (Kandy)": "Pallekele Stadium",
    "Pallekele International Stadium": "Pallekele Stadium",
    "Pallekele Stadium": "Pallekele Stadium",
    "Dambulla": "Rangiri Dambulla",
    "Dambulla International": "Rangiri Dambulla",
    "Rangiri Dambulla International": "Rangiri Dambulla",
    "Rangiri Dambulla": "Rangiri Dambulla",
    "SSC": "SSC Ground",
    "SSC Ground": "SSC Ground",
}

_training_lock = threading.Lock()
_training_status = {
    "running": False,
    "last_started_at": None,
    "last_finished_at": None,
    "last_exit_code": None,
    "last_message": "Training has not been started from the API.",
}

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(title="LPL 2026 Prediction API", version="2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"]
)

VENUE_COL_MAP = {
    "R. Premadasa Stadium": "premadasa_rating",
    "Pallekele Stadium": "pallekele_rating",
    "Rangiri Dambulla": "dambulla_rating",
    "SSC Ground": "premadasa_rating",
}

# ─── Request / Response Models ────────────────────────────────────────────────
class MatchInput(BaseModel):
    team1: str
    team2: str
    venue: str
    team1_home: bool = True
    dew: bool = False
    match_timing: Optional[str] = None

class LiveInput(BaseModel):
    target: int
    runs_scored: int
    balls_bowled: int
    wickets_lost: int

class MatchupInput(BaseModel):
    matchup_sr: float
    balls_faced: int
    dismissals: int

class PlayerPerfInput(BaseModel):
    player_id: str
    venue: str

class MatchSimInput(BaseModel):
    team1: str
    team2: str
    venue: str
    team1_home: bool = True
    dew: bool = False
    match_timing: Optional[str] = None
    seed: Optional[int] = None

class TacticalPlanInput(BaseModel):
    target_batsman: str
    bowler_name: Optional[str] = None
    bowling_type: Optional[str] = None
    venue: str = "R. Premadasa Stadium"
    match_phase: str = "Powerplay"
    batter_hand: Optional[str] = None


def _normalise_venue(venue: str) -> str:
    return VENUE_ALIASES.get(venue, venue)


def _clean_value(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _clean_record(record: dict) -> dict:
    return {str(k): _clean_value(v) for k, v in record.items()}


def _clean_text(value, fallback=""):
    value = _clean_value(value)
    if value is None:
        return fallback
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return fallback
    return text


def _records(df):
    if df is None:
        return []
    return [_clean_record(r) for r in df.fillna("").to_dict(orient="records")]


def _safe_float(value, default=0.0):
    try:
        if value in ("", None):
            return float(default)
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return float(default)
        return parsed
    except (TypeError, ValueError):
        return float(default)


def _clamp(value, low, high):
    return max(low, min(high, value))


def _dew_factor_score(value):
    text = _clean_text(value, "Low").lower()
    return {
        "low": 0.2,
        "medium": 0.55,
        "high": 0.85,
    }.get(text, 0.35)


def _pitch_rows():
    pitch_df = _datasets.get("advanced_pitch_data")
    if pitch_df is None or pitch_df.empty:
        return []

    rows = []
    for row in _records(pitch_df):
        venue_name = _normalise_venue(_clean_text(row.get("Ground_Name")))
        if venue_name not in VENUE_METADATA:
            continue
        rows.append({
            "venue_name": venue_name,
            "label": VENUE_METADATA[venue_name]["label"],
            "city": _clean_text(row.get("Location"), VENUE_METADATA[venue_name]["city"]),
            "match_timing": _clean_text(row.get("Match_Timing"), "Day"),
            "toss_winner_choice": _clean_text(row.get("Toss_Winner_Choice"), "Bat First"),
            "avg_first_innings_score": round(_safe_float(row.get("Avg_1st_Innings_Score"), 155), 1),
            "avg_second_innings_score": round(_safe_float(row.get("Avg_2nd_Innings_Score"), 148), 1),
            "safe_score_target": round(_safe_float(row.get("Safe_Score_Target"), 165), 1),
            "pace_advantage_pct": round(_safe_float(row.get("Pace_Advantage_pct"), 50), 1),
            "spin_advantage_pct": round(_safe_float(row.get("Spin_Advantage_pct"), 50), 1),
            "dew_factor": _clean_text(row.get("Dew_Factor"), "Low"),
            "dew_factor_score": round(_dew_factor_score(row.get("Dew_Factor")), 2),
            "bat_first_win_rate": round(_safe_float(row.get("Bat_First_Win_Rate"), 0.5) * 100, 1),
            "pitch_type": _clean_text(row.get("Pitch_Type"), "Balanced"),
            "powerplay_avg_wickets": round(_safe_float(row.get("Powerplay_Avg_Wickets"), 1.3), 1),
            "death_over_avg_wickets": round(_safe_float(row.get("Death_Over_Avg_Wickets"), 2.5), 1),
            "humidity_pct": round(_safe_float(row.get("Humidity_pct"), 75), 1),
            "temperature_c": round(_safe_float(row.get("Temperature_C"), 29), 1),
        })
    return rows


def _venue_profiles():
    rows = _pitch_rows()
    if not rows:
        return []

    profiles = []
    for venue_name in SUPPORTED_VENUE_ORDER:
        venue_rows = [row for row in rows if row["venue_name"] == venue_name]
        if not venue_rows:
            continue
        day_row = next((row for row in venue_rows if row["match_timing"].lower() == "day"), None)
        night_row = next((row for row in venue_rows if row["match_timing"].lower() == "night"), None)
        default_row = night_row or day_row or venue_rows[0]
        profiles.append({
            "venue_name": venue_name,
            "label": VENUE_METADATA[venue_name]["label"],
            "city": default_row["city"],
            "default_match_timing": default_row["match_timing"],
            "pitch_type": default_row["pitch_type"],
            "avg_first_innings_score": round(float(np.mean([row["avg_first_innings_score"] for row in venue_rows])), 1),
            "avg_second_innings_score": round(float(np.mean([row["avg_second_innings_score"] for row in venue_rows])), 1),
            "safe_score_target": round(float(np.mean([row["safe_score_target"] for row in venue_rows])), 1),
            "pace_advantage_pct": round(float(np.mean([row["pace_advantage_pct"] for row in venue_rows])), 1),
            "spin_advantage_pct": round(float(np.mean([row["spin_advantage_pct"] for row in venue_rows])), 1),
            "dew_factor": default_row["dew_factor"],
            "dew_factor_score": default_row["dew_factor_score"],
            "bat_first_win_rate": round(float(np.mean([row["bat_first_win_rate"] for row in venue_rows])), 1),
            "powerplay_avg_wickets": round(float(np.mean([row["powerplay_avg_wickets"] for row in venue_rows])), 1),
            "death_over_avg_wickets": round(float(np.mean([row["death_over_avg_wickets"] for row in venue_rows])), 1),
            "humidity_pct": round(float(np.mean([row["humidity_pct"] for row in venue_rows])), 1),
            "temperature_c": round(float(np.mean([row["temperature_c"] for row in venue_rows])), 1),
            "timings": venue_rows,
        })
    return profiles


def _pitch_profile_for(venue: str, match_timing: Optional[str] = None, dew: bool = False):
    venue_name = _normalise_venue(venue)
    for profile in _venue_profiles():
        if profile["venue_name"] != venue_name:
            continue
        if match_timing:
            wanted = match_timing.strip().lower()
            row = next((item for item in profile["timings"] if item["match_timing"].lower() == wanted), None)
            if row:
                return row
        if dew:
            row = next((item for item in profile["timings"] if item["match_timing"].lower() == "night"), None)
            if row:
                return row
        return next((item for item in profile["timings"] if item["match_timing"].lower() == "day"), None) or profile["timings"][0]
    return None


def _team_composition_rows():
    return _records(_datasets.get("team_composition"))


def _team_cards():
    comp_rows = _team_composition_rows()
    squad_rows = {r["team"]: r for r in _records(_squad_ratings)}
    if comp_rows:
        source_rows = comp_rows
    else:
        source_rows = list(squad_rows.values())

    cards = []
    for row in source_rows:
        team = row.get("team")
        if not team:
            continue
        squad = squad_rows.get(team, {})
        visuals = TEAM_VISUALS.get(team, {})
        bat_display = row.get("batting_strength_score")
        bowl_display = row.get("bowling_strength_score")
        exp_display = row.get("experience_score")
        if bat_display in ("", None):
            bat_display = round(float(squad.get("batting_rating", 0)) * 10)
        if bowl_display in ("", None):
            bowl_display = round(float(squad.get("bowling_rating", 0)) * 10)
        if exp_display in ("", None):
            exp_display = round(float(squad.get("experience_rating", 0)) * 10)
        overall = round((float(bat_display) + float(bowl_display) + float(exp_display)) / 3, 1)
        cards.append({
            "team": team,
            "color": visuals.get("color", "#3b82f6"),
            "accent": visuals.get("accent", "#93c5fd"),
            "emoji": visuals.get("emoji", team.split()[0]),
            "batting_rating": float(squad.get("batting_rating", float(bat_display) / 10)),
            "bowling_rating": float(squad.get("bowling_rating", float(bowl_display) / 10)),
            "experience_rating": float(squad.get("experience_rating", float(exp_display) / 10)),
            "batting_rating_display": int(round(float(bat_display))),
            "bowling_rating_display": int(round(float(bowl_display))),
            "experience_rating_display": int(round(float(exp_display))),
            "overall_rating_display": overall,
            "predicted_rank": int(row.get("predicted_rank") or 99),
            "cup_probability": float(row.get("win_probability_pct") or 0),
            "squad_depth": row.get("squad_depth_score"),
            "balance_score": row.get("balance_score"),
        })
    return sorted(cards, key=lambda r: (r["predicted_rank"], -r["cup_probability"]))


def _standings_payload():
    teams = _team_cards()
    stats = {
        t["team"]: {
            "team": t["team"], "p": 0, "w": 0, "l": 0, "pts": 0,
            "cup": t["cup_probability"], "color": t["color"], "accent": t["accent"],
            "emoji": t["emoji"], "predicted_rank": t["predicted_rank"],
        }
        for t in teams
    }

    h2h = _datasets.get("h2h_matchups")
    if h2h is not None:
        for row in _records(h2h):
            home = row.get("team_home")
            away = row.get("team_away")
            if home not in stats or away not in stats:
                continue
            stats[home]["p"] += 1
            stats[away]["p"] += 1
            predicted = row.get("prediction_winner")
            if predicted not in (home, away):
                predicted = home if float(row.get("home_win_probability") or 0) >= float(row.get("away_win_probability") or 0) else away
            loser = away if predicted == home else home
            stats[predicted]["w"] += 1
            stats[predicted]["pts"] += 2
            stats[loser]["l"] += 1

    if all(v["p"] == 0 for v in stats.values()):
        for team, item in stats.items():
            rank = item["predicted_rank"]
            item["p"] = 4
            item["w"] = max(0, 5 - rank)
            item["l"] = item["p"] - item["w"]
            item["pts"] = item["w"] * 2

    avg_pts = np.mean([v["pts"] for v in stats.values()]) if stats else 0
    avg_cup = np.mean([v["cup"] for v in stats.values()]) if stats else 0
    rows = sorted(stats.values(), key=lambda v: (-v["pts"], -v["cup"], v["predicted_rank"]))
    for idx, row in enumerate(rows, 1):
        row["pos"] = idx
        nrr_val = ((row["pts"] - avg_pts) / 6.0) + ((row["cup"] - avg_cup) / 80.0)
        row["nrr"] = f"{nrr_val:+.2f}"
        row["qualifies"] = idx <= 3
    return rows


def _label_rows():
    labels = _datasets.get("model_training_labels")
    if labels is None:
        return []
    return _records(labels)


def _find_player(name: str) -> dict:
    if _player_master is None or not name:
        return {}
    rows = _player_master[_player_master["player_name"].str.lower() == name.lower()]
    if rows.empty:
        rows = _player_master[_player_master["player_name"].str.lower().str.contains(name.lower(), regex=False)]
    return _clean_record(rows.iloc[0].fillna("").to_dict()) if not rows.empty else {}


def _batting_caps(limit=10):
    if _player_master is None:
        return []
    df = _player_master.copy()
    needed = [
        "avg_estimate", "powerplay_sr_estimate", "middle_over_sr_estimate",
        "death_over_sr_estimate", "batting_power_play_rating",
        "batting_middle_over_rating", "batting_death_over_rating", "impact_weight",
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = 0
        df[col] = np.nan_to_num(df[col].astype(float), nan=0.0)
    df["sr_estimate"] = df[["powerplay_sr_estimate", "middle_over_sr_estimate", "death_over_sr_estimate"]].replace(0, np.nan).mean(axis=1).fillna(120)
    df["bat_skill"] = df[["batting_power_play_rating", "batting_middle_over_rating", "batting_death_over_rating"]].mean(axis=1)
    df["projected_runs"] = (
        df["avg_estimate"] * 6.2 + df["sr_estimate"] * 0.82 + df["bat_skill"] * 11 + df["impact_weight"] * 4
    ).round().clip(80, 480)
    df = df[df["bat_skill"] > 3].sort_values("projected_runs", ascending=False).head(limit)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "name": row["player_name"],
            "team": row["team"],
            "nationality": row.get("nationality", ""),
            "runs": int(row["projected_runs"]),
            "avg": round(float(row["avg_estimate"] or 0), 1),
            "sr": round(float(row["sr_estimate"] or 0), 1),
            "matches": 10 if row.get("team") != "Kandy Royals" else 11,
        })
    return rows


def _bowling_caps(limit=10):
    if _player_master is None:
        return []
    df = _player_master.copy()
    needed = [
        "wickets_per_match_estimate", "economy_middle", "bowling_power_play_rating",
        "bowling_middle_over_rating", "bowling_death_over_rating", "impact_weight",
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = 0
        df[col] = np.nan_to_num(df[col].astype(float), nan=0.0)
    df["bowl_skill"] = df[["bowling_power_play_rating", "bowling_middle_over_rating", "bowling_death_over_rating"]].mean(axis=1)
    df["projected_wickets"] = (
        df["wickets_per_match_estimate"] * 7.5 + df["bowl_skill"] * 1.4 + df["impact_weight"] * 0.6
    ).round().clip(1, 30)
    df = df[df["bowl_skill"] > 3].sort_values("projected_wickets", ascending=False).head(limit)
    rows = []
    for _, row in df.iterrows():
        economy = float(row.get("economy_middle") or 7.5)
        rows.append({
            "name": row["player_name"],
            "team": row["team"],
            "nationality": row.get("nationality", ""),
            "wickets": int(row["projected_wickets"]),
            "avg": round(max(11.0, 31.0 - float(row["bowl_skill"]) * 1.8), 1),
            "economy": round(economy, 2),
            "matches": 10 if row.get("team") != "Kandy Royals" else 11,
        })
    return rows


def _impact_players(limit=5):
    if _player_master is None:
        return []
    df = _player_master.copy()
    for col in ["match_winner_potential", "consistency_score", "fielding_rating", "impact_weight",
                "batting_middle_over_rating", "bowling_middle_over_rating"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = np.nan_to_num(df[col].astype(float), nan=0.0)
    df["impact_score"] = (
        df["match_winner_potential"] * 4 + df["consistency_score"] * 2 +
        df["fielding_rating"] * 1.2 + df["impact_weight"] * 2 +
        df[["batting_middle_over_rating", "bowling_middle_over_rating"]].max(axis=1) * 3
    ).round().clip(0, 100)
    df = df.sort_values("impact_score", ascending=False).head(limit)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "name": row["player_name"],
            "team": row["team"],
            "nationality": row.get("nationality", ""),
            "score": int(row["impact_score"]),
            "detail": f"Bat {row.get('batting_middle_over_rating', 0):.0f} | Bowl {row.get('bowling_middle_over_rating', 0):.0f} | Pressure {row.get('pressure_handling', 0):.0f}",
        })
    return rows


def _awards_payload():
    label_map = {row.get("label_type"): row for row in _label_rows()}
    orange = _batting_caps()
    purple = _bowling_caps()
    impact = _impact_players()

    allrounders = []
    if _player_master is not None:
        ar = _player_master[_player_master["role"].str.contains("All-Rounder|Spin AR|AR", case=False, na=False)].copy()
        if not ar.empty:
            for col in ["batting_middle_over_rating", "bowling_middle_over_rating", "impact_weight"]:
                ar[col] = np.nan_to_num(ar[col].astype(float), nan=0.0)
            ar["score"] = ar["batting_middle_over_rating"] * 4 + ar["bowling_middle_over_rating"] * 4 + ar["impact_weight"] * 2
            allrounders = _records(ar.sort_values("score", ascending=False).head(1))

    emerging = []
    if _player_master is not None:
        em = _player_master[_player_master["category"].str.contains("Emerging", case=False, na=False)].copy()
        if not em.empty:
            for col in ["match_winner_potential", "consistency_score", "fielding_rating"]:
                em[col] = np.nan_to_num(em[col].astype(float), nan=0.0)
            em["score"] = em["match_winner_potential"] * 5 + em["consistency_score"] * 3 + em["fielding_rating"] * 2
            emerging = _records(em.sort_values("score", ascending=False).head(1))

    def card(label, player_name, stat, reason):
        player = _find_player(player_name)
        return {
            "label": label,
            "name": player_name,
            "team": player.get("team", ""),
            "nationality": player.get("nationality", ""),
            "stat": stat,
            "why": reason,
        }

    pot_name = label_map.get("Player of Tournament", {}).get("label_value") or (impact[0]["name"] if impact else "")
    cards = [
        card("Player of Tournament", pot_name, "Highest combined impact score", label_map.get("Player of Tournament", {}).get("basis", "")),
    ]
    if orange:
        cards.append(card("Best Batsman", orange[0]["name"], f"{orange[0]['runs']} projected runs", "Top modelled run scorer from batting profile and venue suitability."))
    if purple:
        cards.append(card("Best Bowler", purple[0]["name"], f"{purple[0]['wickets']} projected wickets", "Top modelled wicket-taker from bowling phase ratings."))
    if allrounders:
        ar = allrounders[0]
        cards.append(card("Best All-Rounder", ar["player_name"], "Balanced batting and bowling impact", "Highest all-rounder score from connected player data."))
    if emerging:
        em = emerging[0]
        cards.append(card("Best Emerging U23", em["player_name"], "Highest emerging-player impact", "Best U23 blend of potential, consistency, and fielding."))

    return {
        "labels": _label_rows(),
        "orange": orange,
        "purple": purple,
        "top_impact": impact,
        "cards": cards,
    }


def _default_positions(hand: str, style: str):
    return _position_template_for_venue(hand, style, None)


def _position_template_for_venue(hand: str, style: str, venue_profile: Optional[dict]):
    mirror = -1 if hand == "LHB" else 1
    style_l = (style or "").lower()
    pitch_type = _clean_text((venue_profile or {}).get("pitch_type"), "balanced").lower()
    spin_adv = _safe_float((venue_profile or {}).get("spin_advantage_pct"), 50)
    pace_adv = _safe_float((venue_profile or {}).get("pace_advantage_pct"), 50)
    spin_edge = spin_adv - pace_adv
    pace_edge = pace_adv - spin_adv

    if spin_edge >= 12 or any(token in pitch_type for token in ["spin", "slow", "turn", "dry"]):
        base = [
            ("WK", "wk", 0.04, 0.24),
            ("Slip", "close", 0.18, 0.40),
            ("Leg Slip", "close", -0.18, 0.32),
            ("Point", "ring", 0.70, 0.00),
            ("Short Cover", "ring", 0.49, -0.31),
            ("Mid Off", "ring", 0.22, -0.66),
            ("Short Midwicket", "ring", -0.49, -0.22),
            ("Square Leg", "ring", -0.68, 0.02),
            ("Long On", "deep", -0.28, -0.89),
            ("Deep Midwicket", "deep", -0.72, 0.44),
            ("Long Off", "deep", 0.41, -0.90),
        ]
    elif pace_edge >= 12 or any(token in pitch_type for token in ["pace", "bounce", "bouncy"]):
        base = [
            ("WK", "wk", 0.05, 0.25),
            ("2nd Slip", "close", 0.28, 0.42),
            ("Gully", "close", 0.48, 0.23),
            ("Backward Point", "ring", 0.70, 0.08),
            ("Extra Cover", "ring", 0.56, -0.34),
            ("Mid Off", "ring", 0.26, -0.70),
            ("Mid On", "ring", -0.20, -0.72),
            ("Square Leg", "ring", -0.66, 0.08),
            ("Fine Leg", "deep", -0.22, 0.94),
            ("3rd Man", "deep", 0.36, 0.94),
            ("Deep Point", "deep", 0.88, 0.06),
        ]
    else:
        base = [
            ("WK", "wk", 0.05, 0.24),
            ("1st Slip", "close", 0.22, 0.39),
            ("Point", "ring", 0.74, 0.02),
            ("Cover", "ring", 0.58, -0.38),
            ("Mid Off", "ring", 0.28, -0.73),
            ("Mid On", "ring", -0.26, -0.73),
            ("Mid Wkt", "ring", -0.58, -0.32),
            ("Sq Leg", "ring", -0.72, 0.05),
            ("Fine Leg", "deep", -0.26, 0.90),
            ("3rd Man", "deep", 0.30, 0.90),
            ("Long Off", "deep", 0.53, -0.88),
        ]

    power_hitter = any(token in style_l for token in ["aggressive", "explosive", "power", "hitter"])
    anchor = any(token in style_l for token in ["anchor", "solid", "composed"])
    boundary_bias = 0.06 if _safe_float((venue_profile or {}).get("safe_score_target"), 165) >= 178 else 0.0
    boundary_bias += 0.05 if _safe_float((venue_profile or {}).get("dew_factor_score"), 0.2) >= 0.55 else 0.0
    wicket_bias = 0.04 if _safe_float((venue_profile or {}).get("powerplay_avg_wickets"), 1.2) >= 1.5 else 0.0
    slow_bias = 0.05 if spin_edge > 0 else 0.0

    adjusted = []
    for name, role, x, z in base:
        if role == "deep":
            z += boundary_bias
            x *= 1.03 if pace_edge > 0 else 0.98
        elif role == "ring" and slow_bias:
            x *= 0.94
            z *= 0.95
        elif role == "close":
            z += wicket_bias

        if power_hitter and role == "ring" and name in {"Sq Leg", "Square Leg", "Mid Wkt", "Short Midwicket"}:
            name = "Deep Square"
            role = "deep"
            x = -0.82
            z = 0.30 + boundary_bias
        elif anchor and role == "deep" and name == "3rd Man":
            name = "Sweeper Cover"
            role = "ring"
            x = 0.60
            z = -0.08

        if _safe_float((venue_profile or {}).get("dew_factor_score"), 0.2) >= 0.55 and role == "close" and "Slip" in name:
            name = "45 Up"
            role = "ring"
            x = 0.16
            z = 0.56

        adjusted.append({
            "n": name,
            "t": role,
            "x": round(_clamp(x * mirror, -0.92, 0.92), 3),
            "z": round(_clamp(z, -0.94, 0.97), 3),
        })
    return adjusted


def _field_plan_for_player(row: dict, venue_profile: dict):
    batting_hand_text = _clean_text(row.get("batting_style", row.get("batting_style_x", ""))).lower()
    hand = "LHB" if "left" in batting_hand_text else "RHB"
    profile_style = _clean_text(
        row.get("batting_profile_style", row.get("batting_style_y", "")),
        _clean_text(row.get("role", ""), "Balanced batter"),
    )
    weakness = _clean_text(row.get("weakness", ""), "main scoring zones")
    best_type = _clean_text(row.get("best_match_type", ""), "balanced")

    spin_adv = _safe_float(venue_profile.get("spin_advantage_pct"), 50)
    pace_adv = _safe_float(venue_profile.get("pace_advantage_pct"), 50)
    if spin_adv - pace_adv >= 12:
        strategy = "Spin squeeze"
    elif pace_adv - spin_adv >= 12:
        strategy = "Pace cut-off"
    else:
        strategy = "Balanced ring"

    field_type = (
        f"{strategy} setup on {venue_profile.get('pitch_type', 'balanced pitch')} surface "
        f"to close off {weakness}."
    )
    threat = (
        f"{profile_style}; {venue_profile.get('match_timing', 'Day')} game; "
        f"best match type: {best_type}"
    )

    return {
        "venue": venue_profile["venue_name"],
        "label": venue_profile["label"],
        "matchTiming": venue_profile["match_timing"],
        "pitchType": venue_profile["pitch_type"],
        "summary": (
            f"Safe target {int(round(_safe_float(venue_profile.get('safe_score_target'), 165)))} | "
            f"Spin {int(round(spin_adv))}% | Pace {int(round(pace_adv))}% | "
            f"Dew {venue_profile.get('dew_factor', 'Low')}"
        ),
        "threat": threat,
        "fieldType": field_type,
        "pos": _position_template_for_venue(hand, profile_style, venue_profile),
    }


def _fielding_payload():
    if _player_master is None:
        return []
    df = _player_master.copy()
    venue_profiles = _venue_profiles()
    for col in [
        "batting_position", "batting_middle_over_rating", "batting_power_play_rating",
        "batting_death_over_rating", "impact_weight", "match_winner_potential", "fielding_rating"
    ]:
        if col not in df.columns:
            df[col] = 0
        df[col] = np.nan_to_num(df[col].astype(float), nan=0.0)
    df["field_bat_score"] = (
        df["batting_power_play_rating"] * 1.8 +
        df["batting_middle_over_rating"] * 1.6 +
        df["batting_death_over_rating"] * 1.4 +
        df["impact_weight"] * 1.4 +
        df["match_winner_potential"] * 1.2 +
        df["fielding_rating"] * 0.4
    )

    teams = []
    for team in sorted(df["team"].dropna().unique()):
        team_df = df[df["team"] == team].copy()
        batters = team_df.sort_values(
            ["field_bat_score", "batting_position"], ascending=[False, True]
        ).head(4)
        visuals = TEAM_VISUALS.get(team, {})
        items = []
        for _, row in batters.iterrows():
            plans = {
                profile["venue_name"]: _field_plan_for_player(row, _pitch_profile_for(profile["venue_name"]) or profile["timings"][0])
                for profile in venue_profiles
            }
            default_plan = next(iter(plans.values()), {
                "threat": _clean_text(row.get("role", ""), "Balanced batter"),
                "fieldType": "Balanced field setup",
                "pos": _default_positions("RHB", _clean_text(row.get("role", ""), "Balanced batter")),
            })
            items.append({
                "name": _clean_text(row["player_name"]),
                "nationality": _clean_text(row.get("nationality", "")),
                "hand": "LHB" if "left" in _clean_text(row.get("batting_style", ""), "").lower() else "RHB",
                "style": _clean_text(row.get("batting_profile_style", row.get("role", "")), _clean_text(row.get("role", ""), "Balanced batter")),
                "threat": default_plan["threat"],
                "fieldType": default_plan["fieldType"],
                "summary": default_plan.get("summary", ""),
                "modelScore": round(_safe_float(row.get("field_bat_score"), 0), 1),
                "plans": plans,
                "pos": default_plan["pos"],
            })
        teams.append({
            "id": team.lower().replace(" ", "-"),
            "name": team,
            "short": team.split()[0],
            "color": visuals.get("color", "#1a6ec7"),
            "accent": visuals.get("accent", "#4a9ae8"),
            "batsmen": items,
        })
    return teams


def _simulate_overs(total: int, wickets: int, seed_value: str):
    rng = random.Random(seed_value)
    weights = []
    for over in range(20):
        if over < 6:
            base = rng.uniform(0.9, 1.35)
        elif over < 15:
            base = rng.uniform(0.75, 1.12)
        else:
            base = rng.uniform(1.05, 1.55)
        weights.append(base)
    raw = [max(1, int(round(total * w / sum(weights)))) for w in weights]
    diff = total - sum(raw)
    while diff != 0:
        idx = rng.randrange(20)
        if diff > 0:
            raw[idx] += 1
            diff -= 1
        elif raw[idx] > 1:
            raw[idx] -= 1
            diff += 1
    wicket_overs = set(rng.sample(range(20), min(max(int(wickets), 0), 10)))
    cumulative_wickets = 0
    overs = []
    for idx, runs in enumerate(raw):
        wicket = idx in wicket_overs
        if wicket:
            cumulative_wickets += 1
        overs.append({"over": idx + 1, "runs": runs, "wicket": wicket, "cumulative_wickets": cumulative_wickets})
    return overs


def _simulate_match_payload(inp: MatchSimInput):
    if inp.team1 == inp.team2:
        raise HTTPException(400, "Select two different teams")
    venue = _normalise_venue(inp.venue)
    prediction = predict_winner(MatchInput(
        team1=inp.team1, team2=inp.team2, venue=venue,
        team1_home=inp.team1_home, dew=inp.dew,
        match_timing=inp.match_timing,
    ))
    nn = prediction.get("pytorch_nn", {})
    wp1 = nn.get("win_prob_team1")
    if wp1 is None:
        wp1 = prediction.get("logistic_regression", {}).get("win_prob_team1") or 50.0

    seed = inp.seed if inp.seed is not None else f"{inp.team1}|{inp.team2}|{venue}|{datetime.utcnow().date()}"
    rng = random.Random(str(seed))
    score1 = int(round(nn.get("score_team1") or (135 + rng.random() * 35)))
    score2 = int(round(nn.get("score_team2") or (132 + rng.random() * 35)))
    pitch_profile = prediction.get("pitch_profile") or _pitch_profile_for(venue, inp.match_timing, inp.dew)
    if pitch_profile is not None:
        score1 = int(round(score1 * 0.58 + _safe_float(pitch_profile.get("avg_first_innings_score"), 155) * 0.42))
        score2 = int(round(score2 * 0.58 + _safe_float(pitch_profile.get("avg_second_innings_score"), 148) * 0.42))
        if _safe_float(pitch_profile.get("avg_second_innings_score"), 148) > _safe_float(pitch_profile.get("avg_first_innings_score"), 155):
            score2 += rng.randint(2, 8)
        safe_target = _safe_float(pitch_profile.get("safe_score_target"), 165)
        if wp1 >= 50 and score1 < safe_target and rng.random() > 0.45:
            score1 += rng.randint(1, max(3, int((safe_target - score1) * 0.35)))
        elif wp1 < 50 and score2 < safe_target and rng.random() > 0.45:
            score2 += rng.randint(1, max(3, int((safe_target - score2) * 0.35)))
    score1 = max(95, min(235, score1 + rng.randint(-8, 8)))
    score2 = max(90, min(235, score2 + rng.randint(-8, 8)))
    wickets1 = int(round(nn.get("wickets_team1") or rng.randint(4, 8)))
    wickets2 = int(round(nn.get("wickets_team2") or rng.randint(4, 8)))
    if pitch_profile is not None:
        wicket_base = _safe_float(pitch_profile.get("powerplay_avg_wickets"), 1.2) + _safe_float(pitch_profile.get("death_over_avg_wickets"), 2.4)
        wickets1 = int(round(wickets1 * 0.55 + wicket_base * 0.45 + rng.uniform(-0.7, 0.7)))
        wickets2 = int(round(wickets2 * 0.55 + wicket_base * 0.45 + rng.uniform(-0.7, 0.7)))
    wickets1 = int(_clamp(wickets1, 2, 10))
    wickets2 = int(_clamp(wickets2, 2, 10))

    winner = inp.team1 if wp1 >= 50 else inp.team2
    if winner == inp.team1 and score2 >= score1:
        score2 = max(80, score1 - rng.randint(3, 18))
    if winner == inp.team2 and score2 <= score1:
        score2 = score1 + rng.randint(1, 12)

    margin = f"{score1 - score2} runs" if winner == inp.team1 else f"{max(1, 10 - wickets2)} wickets"
    return {
        "team1": inp.team1,
        "team2": inp.team2,
        "venue": venue,
        "match_timing": pitch_profile.get("match_timing") if pitch_profile else inp.match_timing,
        "winner": winner,
        "margin": margin,
        "prediction": prediction,
        "pitch_profile": pitch_profile,
        "innings": [
            {"team": inp.team1, "runs": score1, "wickets": wickets1, "overs": 20, "over_breakdown": _simulate_overs(score1, wickets1, f"{seed}|1")},
            {"team": inp.team2, "runs": score2, "wickets": wickets2, "overs": 20, "over_breakdown": _simulate_overs(score2, wickets2, f"{seed}|2")},
        ],
    }


def _model_status_payload():
    model_files = [
        "pytorch_mtdnn.pth", "pytorch_ppnn.pth", "norm_params.json",
        "ppnn_norm_params.json", "lr_winner.pkl", "rf_score.pkl",
        "poisson_wickets.pkl", "training_history.json",
    ]
    files = []
    for name in model_files:
        path = os.path.join(MODELS_DIR, name)
        files.append({
            "name": name,
            "exists": os.path.exists(path),
            "size_bytes": os.path.getsize(path) if os.path.exists(path) else 0,
            "modified_at": datetime.fromtimestamp(os.path.getmtime(path)).isoformat() if os.path.exists(path) else None,
        })
    return {
        "models_dir": MODELS_DIR,
        "datasets_loaded": {k: (None if v is None else {"rows": int(v.shape[0]), "columns": int(v.shape[1])}) for k, v in _datasets.items()},
        "pytorch_match_model_loaded": _pytorch_model is not None,
        "pytorch_player_model_available": os.path.exists(os.path.join(MODELS_DIR, "pytorch_ppnn.pth")),
        "training_status": _training_status,
        "files": files,
    }


def _run_training_job():
    global _pytorch_model, _norm_params, _lr_winner, _feat_win, _rf_score, _feat_score
    with _training_lock:
        _training_status.update({
            "running": True,
            "last_started_at": datetime.utcnow().isoformat(),
            "last_finished_at": None,
            "last_exit_code": None,
            "last_message": "Training pipeline is running.",
        })
    try:
        proc = subprocess.run(
            [sys.executable, os.path.join(ROOT, "run_pipeline.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60 * 20,
        )
        _pytorch_model, _norm_params = load_pytorch_model(MODELS_DIR)
        _lr_winner = _load_pkl("lr_winner.pkl")
        _feat_win = _load_pkl("feature_names_win.pkl")
        _rf_score = _load_pkl("rf_score.pkl")
        _feat_score = _load_pkl("feature_names_score.pkl")
        message = "Training completed successfully." if proc.returncode == 0 else (proc.stderr[-2000:] or proc.stdout[-2000:])
        with _training_lock:
            _training_status.update({
                "running": False,
                "last_finished_at": datetime.utcnow().isoformat(),
                "last_exit_code": proc.returncode,
                "last_message": message,
            })
    except Exception as exc:
        with _training_lock:
            _training_status.update({
                "running": False,
                "last_finished_at": datetime.utcnow().isoformat(),
                "last_exit_code": -1,
                "last_message": str(exc),
            })

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Serve the creative home page if present, else a JSON status."""
    path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"status": "LPL 2026 Prediction API is running", "version": "2.0"}


@app.get("/index.html")
def index_html():
    path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(path):
        raise HTTPException(404, "frontend/index.html not found")
    return FileResponse(path)


@app.get("/squads")
def get_squads():
    """Return aggregated squad ratings for all 5 teams."""
    if _squad_ratings is None:
        raise HTTPException(503, "Squad ratings not available")
    return _records(_squad_ratings)


@app.get("/teams")
def get_teams():
    if _squad_ratings is None:
        raise HTTPException(503, "Data not available")
    return sorted(_squad_ratings["team"].tolist())


@app.get("/venues")
def get_venues():
    profiles = _venue_profiles()
    if profiles:
        return {"venues": [profile["venue_name"] for profile in profiles], "profiles": profiles}

    venue_df = _datasets.get("venue_conditions")
    if venue_df is None:
        return {"venues": SUPPORTED_VENUE_ORDER}
    venues = [name for name in venue_df["venue_name"].tolist() if "Galle" not in str(name)]
    return {"venues": venues}


@app.get("/players/{team}")
def get_players_by_team(team: str):
    """Return all player profiles for a given team (all data merged)."""
    if _player_master is None:
        raise HTTPException(503, "Player master not built")
    team_df = _player_master[_player_master["team"] == team]
    if team_df.empty:
        raise HTTPException(404, f"Team '{team}' not found")
    # Return a clean subset of columns
    cols = ["player_id", "player_name", "team", "role", "category",
            "batting_style", "bowling_style", "nationality",
            "batting_power_play_rating", "batting_middle_over_rating", "batting_death_over_rating",
            "bowling_power_play_rating", "bowling_middle_over_rating", "bowling_death_over_rating",
            "fielding_rating", "pressure_handling", "consistency_score",
            "premadasa_rating", "galle_rating", "pallekele_rating", "dambulla_rating",
            "best_venue", "worst_venue",
            "powerplay_sr_estimate", "middle_over_sr_estimate", "death_over_sr_estimate",
            "avg_estimate", "spin_handling", "pace_handling", "pressure_avg",
            "wickets_per_match_estimate", "economy_middle",
            "impact_weight", "exp_numeric", "t20_specialist_flag"]
    existing = [c for c in cols if c in team_df.columns]
    return _records(team_df[existing])


_PLAYER_BROWSE_COLS = [
    "player_id", "player_name", "team", "role", "category",
    "batting_style", "bowling_style", "nationality", "age_group",
    "batting_power_play_rating", "batting_middle_over_rating", "batting_death_over_rating",
    "bowling_power_play_rating", "bowling_middle_over_rating", "bowling_death_over_rating",
    "fielding_rating", "pressure_handling", "consistency_score",
    "premadasa_rating", "galle_rating", "pallekele_rating", "dambulla_rating",
    "best_venue", "worst_venue",
    "powerplay_sr_estimate", "middle_over_sr_estimate", "death_over_sr_estimate",
    "avg_estimate", "spin_handling", "pace_handling", "pressure_avg",
    "wickets_per_match_estimate", "economy_middle",
    "impact_weight", "exp_numeric", "match_winner_potential",
]


@app.get("/api/players")
def get_all_players():
    """Return every player (browsable subset) across all teams in one call."""
    if _player_master is None:
        raise HTTPException(503, "Player master not built")
    existing = [c for c in _PLAYER_BROWSE_COLS if c in _player_master.columns]
    players = _records(_player_master[existing])
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "total": len(players),
        "teams": sorted(_player_master["team"].dropna().unique().tolist()),
        "team_visuals": TEAM_VISUALS,
        "players": players,
    }


@app.get("/player/{player_id}")
def get_player_detail(player_id: str):
    """Return full merged profile for a single player_id."""
    if _player_master is None:
        raise HTTPException(503, "Player master not built")
    row = _player_master[_player_master["player_id"] == player_id]
    if row.empty:
        raise HTTPException(404, f"Player {player_id} not found")
    return _clean_record(row.iloc[0].fillna("").to_dict())


@app.get("/player/{player_id}/h2h")
def get_player_h2h(player_id: str):
    """Return head-to-head matchup records for a given player (as batsman or bowler)."""
    h2h_df = _datasets.get("h2h_matchup_advantage")
    master  = _player_master
    if h2h_df is None or master is None:
        raise HTTPException(503, "H2H data not available")
    row = master[master["player_id"] == player_id]
    if row.empty:
        raise HTTPException(404, f"Player {player_id} not found")
    name = row.iloc[0]["player_name"]
    # Batsman records
    as_bat  = h2h_df[h2h_df["batsman"] == name]
    # Bowler records
    as_bowl = h2h_df[h2h_df["bowler"] == name]
    return {"as_batsman": _records(as_bat), "as_bowler": _records(as_bowl)}


@app.post("/predict/winner")
def predict_winner(inp: MatchInput):
    """
    Predict match outcome using PyTorch MT-DNN.
    Returns win probability, scores, wickets, and XAI attributions.
    """
    if _squad_ratings is None:
        raise HTTPException(503, "Squad data not available")

    t1_row = _squad_ratings[_squad_ratings["team"] == inp.team1]
    t2_row = _squad_ratings[_squad_ratings["team"] == inp.team2]
    if t1_row.empty or t2_row.empty:
        raise HTTPException(404, "One or both teams not found")

    t1 = t1_row.iloc[0]
    t2 = t2_row.iloc[0]

    venue_name = _normalise_venue(inp.venue)
    pitch_profile = _pitch_profile_for(venue_name, inp.match_timing, inp.dew)
    v_bat_mod, v_spin_mod, v_pace_mod = 1.0, 1.0, 1.0
    dew_val = 1 if inp.dew else 0

    if pitch_profile is not None:
        avg_score = (
            _safe_float(pitch_profile.get("avg_first_innings_score"), 155)
            + _safe_float(pitch_profile.get("avg_second_innings_score"), 148)
        ) / 2.0
        v_bat_mod = round(_clamp(avg_score / 160.0, 0.82, 1.22), 3)
        v_spin_mod = round(_clamp(_safe_float(pitch_profile.get("spin_advantage_pct"), 50) / 50.0, 0.78, 1.28), 3)
        v_pace_mod = round(_clamp(_safe_float(pitch_profile.get("pace_advantage_pct"), 50) / 50.0, 0.78, 1.28), 3)
        if not inp.dew and _safe_float(pitch_profile.get("dew_factor_score"), 0.2) >= 0.6:
            dew_val = 1
    else:
        venue_df = _datasets.get("venue_conditions")
        if venue_df is not None:
            v_row = venue_df[venue_df["venue_name"] == venue_name]
            if not v_row.empty:
                vr = v_row.iloc[0]
                try:
                    v_bat_mod = float(vr.get("venue_bat_modifier", 1.0))
                    v_spin_mod = float(vr.get("spin_effectiveness", 7.0)) / 7.0
                    v_pace_mod = float(vr.get("pace_effectiveness", 7.0)) / 7.0
                except Exception:
                    pass

    t1_home = int(inp.team1_home)
    t2_home = int(not inp.team1_home)

    feature_dict = {
        "team1_bat_rating":    float(t1["batting_rating"]),
        "team1_bowl_rating":   float(t1["bowling_rating"]),
        "team2_bat_rating":    float(t2["batting_rating"]),
        "team2_bowl_rating":   float(t2["bowling_rating"]),
        "team1_is_home":       t1_home,
        "team2_is_home":       t2_home,
        "venue_bat_modifier":  v_bat_mod,
        "venue_spin_modifier": v_spin_mod,
        "venue_pace_modifier": v_pace_mod,
        "venue_dew_factor":    dew_val,
        "team1_exp_rating":    float(t1["experience_rating"]),
        "team2_exp_rating":    float(t2["experience_rating"]),
        "bat_rating_diff":     float(t1["batting_rating"]) - float(t2["batting_rating"]),
        "bowl_rating_diff":    float(t1["bowling_rating"]) - float(t2["bowling_rating"]),
        "exp_rating_diff":     float(t1["experience_rating"]) - float(t2["experience_rating"]),
        "home_advantage_diff": t1_home - t2_home,
    }

    # PyTorch MT-DNN prediction
    nn_wp = None
    nn_scores = None
    nn_wkts = None
    attributions = {}
    if _pytorch_model and _norm_params:
        nn_wp, nn_scores, nn_wkts, attributions = predict_with_xai(
            feature_dict, model=_pytorch_model, norm_params=_norm_params
        )

    # Classical LR fallback
    lr_wp = None
    if _lr_winner and _feat_win:
        import pandas as pd
        df_in = pd.DataFrame([feature_dict])[_feat_win]
        lr_wp = float(_lr_winner.predict_proba(df_in)[0][1]) * 100.0

    # Classical RF score
    rf_score_t1 = None
    if _rf_score and _feat_score:
        import pandas as pd
        score_inp = {
            "team1_bat_rating": feature_dict["team1_bat_rating"],
            "team2_bowl_rating": feature_dict["team2_bowl_rating"],
            "venue_bat_modifier": v_bat_mod,
            "venue_dew_factor": dew_val,
            "team1_exp_rating": feature_dict["team1_exp_rating"],
            "team2_exp_rating": feature_dict["team2_exp_rating"],
        }
        df_s = pd.DataFrame([score_inp])[_feat_score]
        rf_score_t1 = float(_rf_score.predict(df_s)[0])

    return {
        "team1": inp.team1,
        "team2": inp.team2,
        "venue": venue_name,
        "match_timing": pitch_profile.get("match_timing") if pitch_profile else inp.match_timing,
        "pytorch_nn": {
            "win_prob_team1": round(nn_wp, 2) if nn_wp is not None else None,
            "win_prob_team2": round(100 - nn_wp, 2) if nn_wp is not None else None,
            "score_team1": round(nn_scores[0], 1) if nn_scores else None,
            "score_team2": round(nn_scores[1], 1) if nn_scores else None,
            "wickets_team1": round(nn_wkts[0], 1) if nn_wkts else None,
            "wickets_team2": round(nn_wkts[1], 1) if nn_wkts else None,
        },
        "logistic_regression": {
            "win_prob_team1": round(lr_wp, 2) if lr_wp is not None else None,
            "win_prob_team2": round(100 - lr_wp, 2) if lr_wp is not None else None,
            "score_team1": round(rf_score_t1, 1) if rf_score_t1 else None,
        },
        "xai_attributions": {k: round(v, 5) for k, v in (attributions or {}).items()},
        "feature_input": feature_dict,
        "pitch_profile": pitch_profile,
    }


@app.post("/predict/live")
def predict_live(inp: LiveInput):
    """Live chase calculator: win probability and pressure index."""
    wp = calculate_win_probability(inp.runs_scored, inp.balls_bowled,
                                   inp.target, inp.wickets_lost, chasing=True)
    pi = calculate_pressure_index(inp.runs_scored, inp.balls_bowled,
                                  inp.target, inp.wickets_lost)
    if pi < 35:
        level = "Low"
    elif pi < 60:
        level = "Moderate"
    elif pi < 80:
        level = "High"
    else:
        level = "Extreme"

    return {
        "win_probability_chasing": round(wp, 2),
        "win_probability_defending": round(100 - wp, 2),
        "pressure_index": round(pi, 2),
        "pressure_level": level,
    }


@app.post("/predict/matchup")
def predict_matchup(inp: MatchupInput):
    """Calculate H2H matchup advantage score."""
    score = calculate_matchup_advantage(inp.matchup_sr, inp.dismissals, inp.balls_faced)
    if score >= 100:
        favors = "Batsman"
    elif score >= 80:
        favors = "Even"
    else:
        favors = "Bowler"
    return {"advantage_score": round(score, 2), "favors": favors}


@app.post("/predict/player-performance")
def predict_player_perf(inp: PlayerPerfInput):
    """Run PP-NN for a specific player at a specific venue."""
    if _player_master is None:
        raise HTTPException(503, "Player master not built")
    row = _player_master[_player_master["player_id"] == inp.player_id]
    if row.empty:
        raise HTTPException(404, f"Player {inp.player_id} not found")
    venue = _normalise_venue(inp.venue)
    result = predict_player_performance(row.iloc[0].to_dict(), venue, MODELS_DIR)
    if result is None:
        raise HTTPException(503, "PP-NN model not trained yet")
    return {"player_id": inp.player_id, "venue": venue, **result}


@app.get("/h2h-matchups")
def get_h2h_matchups():
    """Return all head-to-head match predictions."""
    h2h = _datasets.get("h2h_matchups")
    if h2h is None:
        raise HTTPException(503, "H2H data not available")
    return _records(h2h)


@app.get("/tournament-features")
def get_tournament_features():
    feat_df = _datasets.get("tournament_prediction_features")
    if feat_df is None:
        raise HTTPException(503, "Features not available")
    return _records(feat_df)


@app.get("/team-strengths")
def get_team_strengths():
    ts_df = _datasets.get("team_strengths_weaknesses")
    if ts_df is None:
        raise HTTPException(503, "Data not available")
    return _records(ts_df)


@app.get("/api/model-status")
def get_model_status():
    """Return model files, dataset load counts, and training state."""
    return _model_status_payload()


@app.post("/api/train")
def train_models(background_tasks: BackgroundTasks):
    """Start the full training pipeline in the background."""
    if _training_status.get("running"):
        return {"status": "already_running", "training_status": _training_status}
    background_tasks.add_task(_run_training_job)
    return {"status": "started", "message": "Training started. Poll /api/model-status for progress."}


@app.post("/api/train-tactical")
def train_tactical_model():
    """Train only the AI Tactical Engine from the current CSV stack."""
    global _tactical_engine
    try:
        report = train_tactical_engine(ROOT, MODELS_DIR)
        _tactical_engine = _load_pkl("tactical_engine.pkl")
        return {"status": "trained", "report": report}
    except Exception as exc:
        raise HTTPException(500, f"Tactical training failed: {exc}") from exc


@app.get("/api/dashboard")
def get_dashboard_data():
    """Backend data contract for the HTML tournament dashboard."""
    standings = _standings_payload()
    awards = _awards_payload()
    teams = _team_cards()
    champion = next((row.get("label_value") for row in awards["labels"] if row.get("label_type") == "Tournament Winner"), None)
    runner_up = next((row.get("label_value") for row in awards["labels"] if row.get("label_type") == "Runner Up"), None)
    if champion is None and standings:
        champion = standings[0]["team"]
    if runner_up is None and len(standings) > 1:
        runner_up = standings[1]["team"]

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "teams": teams,
        "venues": [profile["venue_name"] for profile in _venue_profiles()] or SUPPORTED_VENUE_ORDER,
        "venue_profiles": _venue_profiles(),
        "standings": standings,
        "bracket": {
            "qualified": standings[:3],
            "champion": champion,
            "runner_up": runner_up,
        },
        "awards": awards,
        "team_strengths": _records(_datasets.get("team_strengths_weaknesses")),
        "h2h_matchups": _records(_datasets.get("h2h_matchups")),
        "model_status": {
            "match_model_loaded": _pytorch_model is not None,
            "player_model_available": os.path.exists(os.path.join(MODELS_DIR, "pytorch_ppnn.pth")),
        },
    }


@app.get("/api/fielding")
def get_fielding_data():
    """Backend data contract for the 3D fielding analyzer HTML."""
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "venues": _venue_profiles(),
        "teams": _fielding_payload(),
    }


@app.post("/api/tactical-plan")
def get_tactical_plan(inp: TacticalPlanInput):
    """AI Tactical Engine: recommend delivery + expected outcome + field coordinates."""
    if _tactical_engine is None:
        raise HTTPException(503, "Tactical engine model is not trained. Run POST /api/train-tactical first.")
    try:
        payload = inp.model_dump()
    except AttributeError:
        payload = inp.dict()
    try:
        return predict_tactical_plan(_tactical_engine, _datasets, payload)
    except Exception as exc:
        raise HTTPException(400, f"Tactical plan failed: {exc}") from exc


@app.post("/api/simulate-match")
def simulate_match(inp: MatchSimInput):
    """Model-backed match simulator for the HTML dashboard."""
    return _simulate_match_payload(inp)


@app.get("/dashboard.html")
def dashboard_html():
    path = os.path.join(FRONTEND_DIR, "dashboard.html")
    if not os.path.exists(path):
        raise HTTPException(404, "frontend/dashboard.html not found")
    return FileResponse(path)


@app.get("/field.html")
def field_html():
    path = os.path.join(FRONTEND_DIR, "field.html")
    if not os.path.exists(path):
        raise HTTPException(404, "frontend/field.html not found")
    return FileResponse(path)


# ─── Research Guide API ───────────────────────────────────────────────────────

def _research_dataset_inventory():
    """Read all CSV metadata: rows, columns, column names, dtypes, nulls."""
    files_map = {
        "readme": "00_README.csv",
        "player_features": "01_player_features.csv",
        "player_master": "01_player_master.csv",
        "player_skill_ratings": "02_player_skill_ratings.csv",
        "team_composition": "03_team_composition_analysis.csv",
        "venue_conditions": "04_venue_ground_conditions.csv",
        "player_venue_suitability": "05_player_venue_suitability.csv",
        "bowling_matchup_matrix": "06_bowling_matchup_matrix.csv",
        "batting_profiles": "07_batting_profiles.csv",
        "category_breakdown": "08_category_breakdown.csv",
        "h2h_matchups": "09_head_to_head_matchups.csv",
        "team_strengths_weaknesses": "10_team_weaknesses_strengths.csv",
        "tournament_prediction_features": "11_tournament_prediction_features.csv",
        "nationality_distribution": "12_nationality_distribution.csv",
        "full_squad": "LPL_2026_Full_Squad.csv",
        "h2h_matchup_advantage": "01_head_to_head_matchup_advantage.csv",
        "win_probability_added": "02_win_probability_added_wpa.csv",
        "pressure_index_reference": "03_pressure_index_reference_grid.csv",
        "match_simulations": "03_match_simulation_dataset (1).csv",
        "model_training_labels": "10_model_training_labels.csv",
        "advanced_pitch_data": "LPL_2026_Advanced_Pitch_Data.csv",
        "schedule": "lpl_schedule_with_grounds (1).csv",
    }

    inventory = []
    total_rows = 0
    total_cols_unique = set()
    for key, filename in files_map.items():
        path = os.path.join(ROOT, filename)
        entry = {
            "key": key,
            "filename": filename,
            "exists": os.path.exists(path),
            "rows": 0,
            "columns": 0,
            "column_names": [],
            "dtypes": {},
            "null_counts": {},
            "null_pct": {},
            "size_bytes": 0,
            "sample_row": None,
            "purpose": "",
        }
        # Describe purpose
        purposes = {
            "readme": "Dataset documentation and column descriptions",
            "player_features": "Player batting/bowling feature ratings (14 features per player)",
            "player_master": "Master player registry — 97 players with roles, nationality, experience",
            "player_skill_ratings": "Detailed phase-wise skill ratings (PP/Middle/Death)",
            "team_composition": "Team composition analysis with strength scores",
            "venue_conditions": "Ground conditions — bat/spin/pace modifiers per venue",
            "player_venue_suitability": "Player venue ratings (1-10 scale) per ground",
            "bowling_matchup_matrix": "Bowler vs RHB/LHB phase matchup data",
            "batting_profiles": "Batting style profiles — SR estimates, boundary %, dot ball %",
            "category_breakdown": "Player category (Star/Icon/Platinum/Gold/Classic) impact weights",
            "h2h_matchups": "Head-to-head match predictions with win probabilities",
            "team_strengths_weaknesses": "Team strategic strengths and weaknesses analysis",
            "tournament_prediction_features": "Tournament-level prediction features",
            "nationality_distribution": "Player nationality breakdown per team",
            "full_squad": "Complete LPL 2026 squad list with player details",
            "h2h_matchup_advantage": "Batsman vs bowler matchup advantage scores",
            "win_probability_added": "Win Probability Added (WPA) reference data",
            "pressure_index_reference": "Pressure Index lookup grid for live predictions",
            "match_simulations": "Match simulation training dataset — 500+ match scenarios",
            "model_training_labels": "Training labels — tournament winner, runner up, awards",
            "advanced_pitch_data": "Advanced pitch analytics per venue per timing",
            "schedule": "LPL 2026 full match schedule with grounds and dates",
        }
        entry["purpose"] = purposes.get(key, "")

        if os.path.exists(path):
            entry["size_bytes"] = os.path.getsize(path)
            try:
                import pandas as pd
                df = pd.read_csv(path)
                entry["rows"] = int(df.shape[0])
                entry["columns"] = int(df.shape[1])
                entry["column_names"] = [str(c).strip() for c in df.columns.tolist()]
                entry["dtypes"] = {str(c): str(df[c].dtype) for c in df.columns}
                entry["null_counts"] = {str(c): int(df[c].isna().sum()) for c in df.columns}
                entry["null_pct"] = {str(c): round(float(df[c].isna().mean()) * 100, 1) for c in df.columns}
                total_rows += entry["rows"]
                total_cols_unique.update(entry["column_names"])
                if not df.empty:
                    entry["sample_row"] = _clean_record(df.iloc[0].fillna("").to_dict())
            except Exception:
                pass
        inventory.append(entry)

    return {
        "total_files": len(inventory),
        "files_found": sum(1 for e in inventory if e["exists"]),
        "files_missing": sum(1 for e in inventory if not e["exists"]),
        "total_data_rows": total_rows,
        "total_unique_columns": len(total_cols_unique),
        "files": inventory,
    }


def _research_model_inventory():
    """Scan models/ directory for all saved model files with metadata."""
    model_info = {
        "pytorch_mtdnn.pth": {
            "type": "PyTorch Multi-Task Deep Neural Network",
            "architecture": "Input(16) → Dense(128) → BN → ReLU → Dropout(0.2) → Dense(64) → BN → ReLU → 3 Heads [Winner(Sigmoid) | Score(Linear×2) | Wickets(Softplus×2)]",
            "input_dim": 16,
            "epochs": 60,
            "optimizer": "Adam (lr=0.003, weight_decay=1e-4)",
            "scheduler": "CosineAnnealingLR",
            "loss": "0.6×BCE(winner) + 0.3×MSE(scores/200) + 0.1×MSE(wickets/10)",
            "task": "Match winner + score + wickets prediction",
        },
        "pytorch_ppnn.pth": {
            "type": "PyTorch Player Performance Neural Network (PP-NN)",
            "architecture": "Input(10) → Dense(64) → BN → ReLU → Dropout(0.2) → Dense(32) → ReLU → [Bat Head(Softplus×2) | Bowl Head(Softplus×2)]",
            "input_dim": 10,
            "epochs": 60,
            "optimizer": "Adam (lr=0.003, weight_decay=1e-4)",
            "loss": "0.5×MSE(batting) + 0.5×MSE(bowling)",
            "task": "Per-player expected runs/SR/wickets/economy at venue",
        },
        "lr_winner.pkl": {
            "type": "Logistic Regression (sklearn)",
            "architecture": "LogisticRegression(C=1.0, max_iter=1000)",
            "input_dim": 16,
            "task": "Binary classification — team1 wins?",
        },
        "rf_score.pkl": {
            "type": "Random Forest Regressor (sklearn)",
            "architecture": "RandomForestRegressor(n_estimators=100, max_depth=5)",
            "input_dim": 6,
            "task": "First innings score regression",
        },
        "poisson_wickets.pkl": {
            "type": "Poisson Regressor (sklearn)",
            "architecture": "PoissonRegressor(alpha=1.0)",
            "input_dim": 5,
            "task": "Wickets per innings count regression",
        },
        "xgb_winner.pkl": {
            "type": "XGBoost Classifier",
            "architecture": "XGBClassifier(optional, may not exist)",
            "input_dim": 16,
            "task": "Match winner classification (boosted trees)",
        },
        "tactical_engine.pkl": {
            "type": "AI Tactical Engine (sklearn pipeline)",
            "architecture": "Multi-class classifier for delivery outcome prediction",
            "input_dim": "variable",
            "task": "Predict delivery outcome (Dot/4/6/Out) + recommend bowling plan",
        },
    }

    models = []
    for name, info in model_info.items():
        path = os.path.join(MODELS_DIR, name)
        models.append({
            "filename": name,
            "exists": os.path.exists(path),
            "size_bytes": os.path.getsize(path) if os.path.exists(path) else 0,
            "size_display": f"{os.path.getsize(path)/1024:.1f} KB" if os.path.exists(path) else "N/A",
            "modified_at": datetime.fromtimestamp(os.path.getmtime(path)).isoformat() if os.path.exists(path) else None,
            **info,
        })

    # Also include support files
    support_files = [
        "norm_params.json", "ppnn_norm_params.json",
        "training_history.json", "tactical_engine_report.json",
        "feature_names_win.pkl", "feature_names_score.pkl", "feature_names_wickets.pkl",
    ]
    support = []
    for name in support_files:
        path = os.path.join(MODELS_DIR, name)
        support.append({
            "filename": name,
            "exists": os.path.exists(path),
            "size_bytes": os.path.getsize(path) if os.path.exists(path) else 0,
        })

    return {
        "total_models": len(models),
        "models_trained": sum(1 for m in models if m["exists"]),
        "models_missing": sum(1 for m in models if not m["exists"]),
        "models": models,
        "support_files": support,
    }


def _research_training_history():
    """Load training_history.json for loss curve data."""
    path = os.path.join(MODELS_DIR, "training_history.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            history = json.load(f)
        losses = [h["train_loss"] for h in history]
        return {
            "epochs": len(history),
            "initial_loss": round(losses[0], 6) if losses else None,
            "final_loss": round(losses[-1], 6) if losses else None,
            "best_loss": round(min(losses), 6) if losses else None,
            "best_epoch": losses.index(min(losses)) + 1 if losses else None,
            "loss_reduction_pct": round((1 - losses[-1] / losses[0]) * 100, 1) if losses and losses[0] > 0 else 0,
            "converged": abs(losses[-1] - losses[-5]) < 0.02 if len(losses) >= 5 else False,
            "curve": [{"epoch": h["epoch"], "loss": round(h["train_loss"], 6)} for h in history],
        }
    except Exception:
        return None


def _research_feature_pipeline():
    """Describe the feature engineering pipeline in detail."""
    # Match features (16-dim vector)
    match_features = [
        {"name": "team1_bat_rating", "group": "Team Batting", "description": "Team 1 aggregated batting rating (top 3 batters avg)", "range": "75-84", "type": "continuous"},
        {"name": "team1_bowl_rating", "group": "Team Bowling", "description": "Team 1 aggregated bowling rating (top 4 bowlers avg)", "range": "78-90", "type": "continuous"},
        {"name": "team2_bat_rating", "group": "Team Batting", "description": "Team 2 aggregated batting rating", "range": "75-84", "type": "continuous"},
        {"name": "team2_bowl_rating", "group": "Team Bowling", "description": "Team 2 aggregated bowling rating", "range": "78-90", "type": "continuous"},
        {"name": "team1_is_home", "group": "Context", "description": "1 if Team 1 is playing at home ground", "range": "0 or 1", "type": "binary"},
        {"name": "team2_is_home", "group": "Context", "description": "1 if Team 2 is playing at home ground", "range": "0 or 1", "type": "binary"},
        {"name": "venue_bat_modifier", "group": "Venue", "description": "Venue batting modifier (avg_score / 160)", "range": "0.82-1.22", "type": "continuous"},
        {"name": "venue_spin_modifier", "group": "Venue", "description": "Venue spin effectiveness factor", "range": "0.78-1.28", "type": "continuous"},
        {"name": "venue_pace_modifier", "group": "Venue", "description": "Venue pace effectiveness factor", "range": "0.78-1.28", "type": "continuous"},
        {"name": "venue_dew_factor", "group": "Venue", "description": "Dew factor (0=none, 1=heavy)", "range": "0 or 1", "type": "binary"},
        {"name": "team1_exp_rating", "group": "Context", "description": "Team 1 experience rating (High=9, Mid=7, Low=5)", "range": "76-88", "type": "continuous"},
        {"name": "team2_exp_rating", "group": "Context", "description": "Team 2 experience rating", "range": "76-88", "type": "continuous"},
        {"name": "bat_rating_diff", "group": "Differential", "description": "Team1 batting - Team2 batting (shows gap)", "range": "-9 to +9", "type": "derived"},
        {"name": "bowl_rating_diff", "group": "Differential", "description": "Team1 bowling - Team2 bowling", "range": "-12 to +12", "type": "derived"},
        {"name": "exp_rating_diff", "group": "Differential", "description": "Team1 experience - Team2 experience", "range": "-12 to +12", "type": "derived"},
        {"name": "home_advantage_diff", "group": "Differential", "description": "Home advantage gap (1=T1 home, -1=T2 home, 0=neutral)", "range": "-1 to +1", "type": "derived"},
    ]

    # Player features (10-dim PP-NN vector)
    player_features = [
        {"name": "batting_power_play_rating", "group": "Batting Phase", "description": "PP batting skill 1-10", "range": "1-10"},
        {"name": "batting_middle_over_rating", "group": "Batting Phase", "description": "Middle overs batting skill", "range": "1-9"},
        {"name": "batting_death_over_rating", "group": "Batting Phase", "description": "Death overs batting skill", "range": "1-9"},
        {"name": "bowling_power_play_rating", "group": "Bowling Phase", "description": "PP bowling skill 0-9", "range": "0-9"},
        {"name": "bowling_middle_over_rating", "group": "Bowling Phase", "description": "Middle overs bowling skill", "range": "0-9"},
        {"name": "bowling_death_over_rating", "group": "Bowling Phase", "description": "Death overs bowling skill", "range": "0-8"},
        {"name": "venue_rating", "group": "Venue", "description": "Player venue suitability (per ground)", "range": "0-9"},
        {"name": "spin_handling", "group": "Batting Skill", "description": "Spin bowling handling ability", "range": "0-9"},
        {"name": "pace_handling", "group": "Batting Skill", "description": "Pace bowling handling ability", "range": "0-9"},
        {"name": "pressure_handling", "group": "Mental", "description": "Performance under pressure", "range": "4-9"},
    ]

    # Normalization params
    norm = None
    norm_path = os.path.join(MODELS_DIR, "norm_params.json")
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            norm = json.load(f)

    ppnn_norm = None
    ppnn_path = os.path.join(MODELS_DIR, "ppnn_norm_params.json")
    if os.path.exists(ppnn_path):
        with open(ppnn_path) as f:
            ppnn_norm = json.load(f)

    # Feature groups summary
    groups = {}
    for f in match_features:
        g = f["group"]
        if g not in groups:
            groups[g] = {"count": 0, "features": []}
        groups[g]["count"] += 1
        groups[g]["features"].append(f["name"])

    return {
        "match_model": {
            "name": "MT-DNN Match Features",
            "total_features": len(match_features),
            "feature_groups": groups,
            "features": match_features,
            "normalization": {
                "method": "Min-Max Scaling (0-1)",
                "params_file": "norm_params.json",
                "feat_min": norm["feat_min"] if norm else None,
                "feat_max": norm["feat_max"] if norm else None,
            },
            "targets": [
                {"name": "team1_won", "type": "binary", "task": "Classification (BCE loss)"},
                {"name": "team1_score / team2_score", "type": "continuous", "task": "Regression (MSE loss, scaled /200)"},
                {"name": "team1_wickets / team2_wickets", "type": "count", "task": "Regression (MSE loss, scaled /10)"},
            ],
        },
        "player_model": {
            "name": "PP-NN Player Features",
            "total_features": len(player_features),
            "features": player_features,
            "normalization": {
                "method": "Min-Max Scaling (0-1)",
                "params_file": "ppnn_norm_params.json",
                "x_min": ppnn_norm["x_min"] if ppnn_norm else None,
                "x_max": ppnn_norm["x_max"] if ppnn_norm else None,
            },
            "targets": [
                {"name": "expected_runs / expected_sr", "type": "continuous", "task": "Batting output (scaled ×200, ×50)"},
                {"name": "expected_wickets / expected_economy", "type": "continuous", "task": "Bowling output (scaled ×5, ×12)"},
            ],
        },
        "data_preprocessing_steps": [
            {"step": 1, "name": "Load CSV Files", "description": "Read all 21 CSV files via LPLDataLoader.load_all_data()", "code": "loader.load_all_data()"},
            {"step": 2, "name": "Strip & Clean", "description": "Strip whitespace from column names and string values", "code": "df.columns = [c.strip() for c in df.columns]"},
            {"step": 3, "name": "Build Player Master", "description": "Merge 7 CSVs (squad + master + skills + venue + bowling + batting + category) into unified player profiles", "code": "build_player_master(datasets)"},
            {"step": 4, "name": "Feature Engineering", "description": "Compute derived features: rating diffs, home advantage, venue modifiers", "code": "fe.prepare_match_features()"},
            {"step": 5, "name": "Handle Missing Values", "description": "Fill NaN with 0.0 for numeric columns, defaults for categorical", "code": "df[numeric_cols] = df[numeric_cols].fillna(0.0)"},
            {"step": 6, "name": "Min-Max Normalize", "description": "Scale all 16 features to [0, 1] range using training set min/max", "code": "X_norm = (X - feat_min) / (feat_max - feat_min)"},
            {"step": 7, "name": "Train/Test Split", "description": "80/20 split with random_state=42 for reproducibility", "code": "random_split(dataset, [0.8, 0.2], seed=42)"},
        ],
        "data_cleaning_report": {
            "missing_value_strategy": "Fill with 0.0 (numeric) or default category (string)",
            "outlier_handling": "Clamped to min/max ranges, no outlier removal",
            "encoding": {
                "experience": "High=9.0, Medium=7.0, Low=5.0",
                "t20_specialist": "yes=1.0, no=0.0",
                "home_advantage": "home=1, away=0",
                "dew_factor": "Low=0.2, Medium=0.55, High=0.85",
            },
        },
    }


def _research_evaluation_metrics():
    """Run quick evaluation on saved models and return metrics."""
    import pandas as pd
    metrics = {
        "match_winner": {},
        "score_regression": {},
        "wickets_regression": {},
        "pytorch_mtdnn": {},
        "tactical_engine": {},
    }

    # Get processed data
    try:
        fe = LPLFeatureEngineer(_datasets)
        processed_df = fe.prepare_match_features()
    except Exception:
        return metrics

    # LR Winner evaluation
    if _lr_winner and _feat_win:
        try:
            from sklearn.metrics import accuracy_score, log_loss
            from sklearn.model_selection import train_test_split
            X_win = processed_df[_feat_win]
            y_win = processed_df['team1_won'].astype(int)
            _, X_te, _, y_te = train_test_split(X_win, y_win, test_size=0.2, random_state=42)
            y_pred = _lr_winner.predict(X_te)
            y_prob = _lr_winner.predict_proba(X_te)[:, 1]
            metrics["match_winner"] = {
                "model": "Logistic Regression",
                "accuracy_pct": round(accuracy_score(y_te, y_pred) * 100, 2),
                "log_loss": round(log_loss(y_te, y_prob), 4),
                "test_size": int(len(X_te)),
                "train_size": int(len(X_win) - len(X_te)),
            }
        except Exception:
            pass

    # RF Score evaluation
    if _rf_score and _feat_score:
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            from sklearn.model_selection import train_test_split
            X_sc = processed_df[_feat_score]
            y_sc = processed_df['team1_score'].astype(float)
            _, X_te, _, y_te = train_test_split(X_sc, y_sc, test_size=0.2, random_state=42)
            y_pred = _rf_score.predict(X_te)
            metrics["score_regression"] = {
                "model": "Random Forest Regressor",
                "rmse": round(float(np.sqrt(mean_squared_error(y_te, y_pred))), 2),
                "mae": round(float(mean_absolute_error(y_te, y_pred)), 2),
                "test_size": int(len(X_te)),
            }
        except Exception:
            pass

    # PyTorch MT-DNN evaluation
    if _pytorch_model and _norm_params:
        try:
            from data_prep import MatchDataset
            from sklearn.metrics import accuracy_score, log_loss
            import torch
            from torch.utils.data import random_split as rs, DataLoader

            dataset = MatchDataset(processed_df, fit=True)
            Xn = dataset.normalize(dataset.X,
                                   np.array(_norm_params["feat_min"], dtype=np.float32),
                                   np.array(_norm_params["feat_max"], dtype=np.float32))
            dataset.X = Xn
            n_train = int(0.8 * len(dataset))
            _, val_ds = rs(dataset, [n_train, len(dataset) - n_train],
                           generator=torch.Generator().manual_seed(42))
            val_loader = DataLoader(val_ds, batch_size=32)

            all_true, all_pred, all_prob = [], [], []
            all_s_true, all_s_pred = [], []
            all_k_true, all_k_pred = [], []

            with torch.no_grad():
                for X_b, y_w, y_s, y_k in val_loader:
                    pw, ps, pk = _pytorch_model(X_b)
                    prob = pw.numpy().flatten()
                    pred = (prob >= 0.5).astype(int)
                    all_true.extend(y_w.numpy().flatten().tolist())
                    all_prob.extend(prob.tolist())
                    all_pred.extend(pred.tolist())
                    all_s_true.extend(y_s.numpy().tolist())
                    all_s_pred.extend((ps.numpy() * 200.0).tolist())
                    all_k_true.extend(y_k.numpy().tolist())
                    all_k_pred.extend((pk.numpy() * 10.0).tolist())

            s_tr = np.array(all_s_true)
            s_pr = np.array(all_s_pred)
            k_tr = np.array(all_k_true)
            k_pr = np.array(all_k_pred)

            metrics["pytorch_mtdnn"] = {
                "model": "PyTorch MT-DNN (Multi-Task)",
                "winner_accuracy_pct": round(accuracy_score(all_true, all_pred) * 100, 2),
                "winner_log_loss": round(log_loss(all_true, all_prob), 4),
                "score_rmse_t1": round(float(np.sqrt(np.mean((s_tr[:, 0] - s_pr[:, 0])**2))), 2),
                "score_mae_t1": round(float(np.mean(np.abs(s_tr[:, 0] - s_pr[:, 0]))), 2),
                "wickets_rmse_t1": round(float(np.sqrt(np.mean((k_tr[:, 0] - k_pr[:, 0])**2))), 2),
                "wickets_mae_t1": round(float(np.mean(np.abs(k_tr[:, 0] - k_pr[:, 0]))), 2),
                "val_size": len(all_true),
            }
        except Exception:
            pass

    # Tactical Engine report
    report_path = os.path.join(MODELS_DIR, "tactical_engine_report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path) as f:
                report = json.load(f)
            metrics["tactical_engine"] = {
                "model": "AI Tactical Engine (RandomForest pipeline)",
                "accuracy_pct": round(float(report.get("accuracy", 0)) * 100, 2),
                "training_rows": report.get("training_rows", 0),
                "test_rows": report.get("test_rows", 0),
                "class_counts": report.get("class_counts", {}),
                "trained_at": report.get("trained_at"),
            }
        except Exception:
            pass

    return metrics


def _research_match_data():
    """Return match-by-match simulation data showing how features differ per match."""
    sim_df = _datasets.get("match_simulations")
    if sim_df is None:
        return []

    matches = []
    for _, row in sim_df.iterrows():
        match = _clean_record(row.fillna("").to_dict())
        # Add derived features
        match["bat_rating_diff"] = round(_safe_float(match.get("team1_bat_rating")) - _safe_float(match.get("team2_bat_rating")), 2)
        match["bowl_rating_diff"] = round(_safe_float(match.get("team1_bowl_rating")) - _safe_float(match.get("team2_bowl_rating")), 2)
        match["exp_rating_diff"] = round(_safe_float(match.get("team1_exp_rating")) - _safe_float(match.get("team2_exp_rating")), 2)
        t1_home = 1 if match.get("team1_is_home") in (1, "1", True, "True", "true") else 0
        t2_home = 1 if match.get("team2_is_home") in (1, "1", True, "True", "true") else 0
        match["home_advantage_diff"] = t1_home - t2_home
        matches.append(match)

    return matches


def _research_algorithms():
    """Describe each algorithm used, with pros/cons/accuracy for cricket prediction."""
    return [
        {
            "name": "Logistic Regression",
            "type": "Classical ML",
            "task": "Match winner classification",
            "icon": "📈",
            "accuracy_range": "55-62%",
            "pros": ["Fast training, interpretable coefficients", "Good baseline — shows which features matter", "Works well on small datasets (< 500 matches)"],
            "cons": ["Linear decision boundary only", "Cannot capture complex feature interactions", "Assumes features are linearly separable"],
            "when_to_use": "Starting point / explainability needed / small data",
            "in_our_system": True,
            "file": "lr_winner.pkl",
        },
        {
            "name": "Random Forest Regressor",
            "type": "Classical ML",
            "task": "First innings score prediction",
            "icon": "🌲",
            "accuracy_range": "RMSE 12-25 runs",
            "pros": ["Handles non-linear relationships", "No feature scaling needed", "Built-in feature importance"],
            "cons": ["Can overfit on small datasets", "Slower inference than linear models", "Doesn't extrapolate well beyond training range"],
            "when_to_use": "Medium dataset (200+ matches), regression tasks",
            "in_our_system": True,
            "file": "rf_score.pkl",
        },
        {
            "name": "Poisson Regressor",
            "type": "Classical ML",
            "task": "Wickets per innings prediction",
            "icon": "🎯",
            "accuracy_range": "RMSE 1.5-2.5 wickets",
            "pros": ["Designed for count data (wickets are counts)", "Assumes positive integer outputs", "Statistically principled for rare events"],
            "cons": ["Assumes mean=variance (may not hold)", "Limited flexibility in decision boundary"],
            "when_to_use": "Count prediction (wickets, boundaries, dot balls)",
            "in_our_system": True,
            "file": "poisson_wickets.pkl",
        },
        {
            "name": "PyTorch MT-DNN",
            "type": "Deep Learning",
            "task": "Multi-task: winner + score + wickets",
            "icon": "🧠",
            "accuracy_range": "57-65% (winner), RMSE 15-22 (score)",
            "pros": ["Learns all 3 tasks simultaneously (shared features)", "Deep non-linear representations", "Gradient×Input XAI for explainability"],
            "cons": ["Needs more data than classical models", "Harder to interpret than LR", "Training requires GPU for speed"],
            "when_to_use": "Production model with enough data (500+ matches)",
            "in_our_system": True,
            "file": "pytorch_mtdnn.pth",
        },
        {
            "name": "Player Performance NN (PP-NN)",
            "type": "Deep Learning",
            "task": "Per-player batting/bowling output prediction",
            "icon": "👤",
            "accuracy_range": "Venue-specific performance",
            "pros": ["Per-player predictions at each venue", "Captures player-venue fit", "Separate batting and bowling heads"],
            "cons": ["Limited by quality of player rating data", "No sequential/form data yet", "Needs real match data for validation"],
            "when_to_use": "Player selection, match-day analysis",
            "in_our_system": True,
            "file": "pytorch_ppnn.pth",
        },
        {
            "name": "AI Tactical Engine",
            "type": "Ensemble ML",
            "task": "Delivery outcome + field placement",
            "icon": "🎲",
            "accuracy_range": "98.9% (on synthetic data)",
            "pros": ["Predicts delivery outcomes (dot/4/6/out)", "Recommends field positions", "Multi-class with confidence scores"],
            "cons": ["Trained on synthetic data, not real ball-by-ball", "98.9% accuracy means overfitting on synthetic labels", "Needs CricSheet ball-by-ball data for validation"],
            "when_to_use": "Tactical bowling plans, field setup optimization",
            "in_our_system": True,
            "file": "tactical_engine.pkl",
        },
        {
            "name": "Gradient Boosting (XGBoost)",
            "type": "Classical ML",
            "task": "Match winner classification",
            "icon": "🚀",
            "accuracy_range": "58-65%",
            "pros": ["Best accuracy on tabular data (competitions)", "Built-in regularization", "Handles missing values natively"],
            "cons": ["Needs hyperparameter tuning (Optuna)", "Slower training than LR", "Can overfit without proper CV"],
            "when_to_use": "Production model, Kaggle-style problems, ≥300 matches",
            "in_our_system": False,
            "file": "xgb_winner.pkl (optional)",
        },
        {
            "name": "Elo Rating System",
            "type": "Statistical",
            "task": "Dynamic team strength tracking",
            "icon": "📊",
            "accuracy_range": "~58%",
            "pros": ["Tracks form over time dynamically", "Simple to implement and update", "No training data needed — updates after each match"],
            "cons": ["No venue/pitch/player info", "Slow to adapt to roster changes", "Only tracks team-level, not player-level"],
            "when_to_use": "Live tournament tracking, team form index",
            "in_our_system": False,
            "file": "N/A (calculate on-the-fly)",
        },
    ]


def _research_pipeline_plan():
    """Return the full 8-week research roadmap and API call structure."""
    return {
        "roadmap": [
            {
                "phase": "Phase 1", "title": "Data Collection", "weeks": "Week 1-2",
                "icon": "📥", "color": "#4361ee",
                "tasks": [
                    "CricSheet CSV download (free) — cricsheet.org/downloads/lpl_csv2.zip",
                    "ESPNCricinfo scraper for LPL match scorecards",
                    "LPL 2020-2025 historical match data (300-500 matches)",
                    "Build SQLite/PostgreSQL database for structured storage",
                ],
                "deliverables": ["data/raw/ directory with all CSVs", "data/db.sqlite with match database"],
            },
            {
                "phase": "Phase 2", "title": "Feature Engineering", "weeks": "Week 3-4",
                "icon": "⚙️", "color": "#7b2d8b",
                "tasks": [
                    "Rolling team stats calculator (last 10 matches window)",
                    "Venue conditions pipeline (pitch, weather, dew)",
                    "H2H win rate tracker per team pair",
                    "Player form index (batting avg, SR, wickets in recent matches)",
                ],
                "deliverables": ["30-dimension feature vector per match", "Feature matrix: (N_matches × 30)"],
            },
            {
                "phase": "Phase 3", "title": "Model Training", "weeks": "Week 5-6",
                "icon": "🤖", "color": "#2a9d8f",
                "tasks": [
                    "Train 3-4 model types (LR, RF, GBM, DNN)",
                    "Cross-validation (5-fold stratified)",
                    "Hyperparameter tuning with Optuna (100 trials)",
                    "Ensemble model building (weighted average of best models)",
                ],
                "deliverables": ["models/ directory with trained .pkl and .pth files", "Best model selected by CV accuracy"],
            },
            {
                "phase": "Phase 4", "title": "Evaluation & Deploy", "weeks": "Week 7-8",
                "icon": "🚀", "color": "#e63946",
                "tasks": [
                    "Confusion matrix + ROC curve + calibration plot",
                    "Brier score + log loss for probability quality",
                    "Flask/FastAPI REST API endpoint deployment",
                    "React/HTML dashboard integration with live predictions",
                ],
                "deliverables": ["Production API at /predict/winner", "Dashboard with real-time predictions"],
            },
        ],
        "tech_stack": {
            "data": ["pandas", "requests", "SQLite", "numpy"],
            "ml": ["scikit-learn", "PyTorch", "xgboost", "optuna"],
            "visualization": ["matplotlib", "seaborn", "plotly"],
            "deployment": ["FastAPI", "uvicorn", "Docker", "Streamlit"],
        },
        "api_endpoints": [
            {"method": "GET", "path": "/", "description": "Health check — API running status"},
            {"method": "GET", "path": "/squads", "description": "Squad ratings for all 5 teams"},
            {"method": "GET", "path": "/teams", "description": "List of team names"},
            {"method": "GET", "path": "/venues", "description": "Venue profiles with pitch data"},
            {"method": "GET", "path": "/players/{team}", "description": "All player profiles for a team"},
            {"method": "GET", "path": "/player/{id}", "description": "Full merged profile for one player"},
            {"method": "GET", "path": "/player/{id}/h2h", "description": "Head-to-head matchup records"},
            {"method": "POST", "path": "/predict/winner", "description": "MT-DNN match outcome prediction + XAI"},
            {"method": "POST", "path": "/predict/live", "description": "Live chase win probability + pressure index"},
            {"method": "POST", "path": "/predict/matchup", "description": "H2H matchup advantage score"},
            {"method": "POST", "path": "/predict/player-performance", "description": "PP-NN player performance at venue"},
            {"method": "GET", "path": "/api/dashboard", "description": "Full tournament dashboard data contract"},
            {"method": "GET", "path": "/api/fielding", "description": "3D fielding analyzer data contract"},
            {"method": "POST", "path": "/api/tactical-plan", "description": "AI tactical bowling plan + field setup"},
            {"method": "POST", "path": "/api/simulate-match", "description": "Full match simulation with over-by-over"},
            {"method": "GET", "path": "/api/model-status", "description": "Model files, dataset counts, training state"},
            {"method": "POST", "path": "/api/train", "description": "Start full training pipeline (background)"},
            {"method": "POST", "path": "/api/train-tactical", "description": "Train tactical engine only"},
            {"method": "GET", "path": "/api/research-guide", "description": "Full research guide data — this endpoint!"},
        ],
        "key_insight": "Cricket prediction accuracy 57-65% ekak NORMAL. NBA/NFL ekath 60% vitharai. Upset always thiyanawa — data eka bayalata pennanna puluwan, gamana guarantee karanna behe. Venue conditions (pace, bounce, dew) MOST important features — team batting/bowling vitharai neme!",
    }


@app.get("/api/research-guide")
def get_research_guide():
    """
    🔬 ML Research Guide — Full pipeline data endpoint.
    Returns: datasets inventory, feature pipeline, model inventory,
    training history, evaluation metrics, match data, algorithms, and roadmap.
    """
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "system_summary": {
            "project": "LPL 2026 Prediction System",
            "version": "2.0",
            "approach": "Multi-Task Deep Neural Network + Classical ML Ensemble",
            "data_type": "Rule-based synthetic (NOT real historical match data)",
            "total_models": 7,
            "total_csv_files": 22,
            "frontend_pages": ["dashboard.html", "field.html", "research.html"],
            "backend_framework": "FastAPI + PyTorch + scikit-learn",
            "current_limitation": "Model is trained on manually-set ratings and synthetic match simulations, NOT on real historical LPL match data. For proper ML, need CricSheet ball-by-ball data.",
        },
        "datasets": _research_dataset_inventory(),
        "feature_pipeline": _research_feature_pipeline(),
        "models": _research_model_inventory(),
        "training_history": _research_training_history(),
        "evaluation": _research_evaluation_metrics(),
        "match_data": _research_match_data(),
        "algorithms": _research_algorithms(),
        "pipeline_plan": _research_pipeline_plan(),
    }


@app.get("/research.html")
def research_html():
    path = os.path.join(FRONTEND_DIR, "research.html")
    if not os.path.exists(path):
        raise HTTPException(404, "frontend/research.html not found")
    return FileResponse(path)


# ─── Tournament Forecast (schedule → predictions + Monte-Carlo + game plans) ──

def _schedule_path():
    """Prefer the advanced schedule file, fall back to the grounds schedule."""
    candidates = ["lpl_schedule_advanced.csv", "lpl_schedule_with_grounds (1).csv",
                  "lpl_schedule_with_grounds.csv"]
    for name in candidates:
        p = os.path.join(ROOT, name)
        if os.path.exists(p):
            return p
    return os.path.join(ROOT, candidates[0])


def _forecast_predict_fn(team1, team2, venue, team1_home, dew, match_timing):
    """Callback for the forecast engine → returns win prob of team1 (0-100)."""
    result = predict_winner(MatchInput(
        team1=team1, team2=team2, venue=venue,
        team1_home=team1_home, dew=dew, match_timing=match_timing,
    ))
    nn = result.get("pytorch_nn", {})
    wp = nn.get("win_prob_team1")
    if wp is None:
        wp = result.get("logistic_regression", {}).get("win_prob_team1")
    return wp if wp is not None else 50.0


def _forecast_pitch_fn(venue, match_timing, dew):
    return _pitch_profile_for(venue, match_timing, dew)


_forecast_cache = {"payload": None, "sims": None}


@app.get("/api/tournament-forecast")
def get_tournament_forecast(sims: int = 5000, refresh: bool = False):
    """
    Full tournament forecast from the fixture schedule:
      • per-match win probabilities (trained match model)
      • Monte-Carlo play-off / final / champion probabilities
      • projected points table
      • AI tactical game plan for every match
    """
    sims = int(max(500, min(20000, sims)))
    if (not refresh) and _forecast_cache["payload"] is not None and _forecast_cache["sims"] == sims:
        return _forecast_cache["payload"]

    schedule_path = _schedule_path()
    if not os.path.exists(schedule_path):
        raise HTTPException(503, "Schedule CSV not found")

    forecast = forecast_engine.build_forecast(
        schedule_path, _forecast_predict_fn, _forecast_pitch_fn, n_sims=sims
    )
    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "schedule_file": os.path.basename(schedule_path),
        "model_loaded": _pytorch_model is not None,
        **forecast,
    }
    payload = json.loads(json.dumps(payload, default=lambda o: None))  # clean NaN/np types
    _forecast_cache.update({"payload": payload, "sims": sims})
    return payload


@app.get("/forecast.html")
def forecast_html():
    path = os.path.join(FRONTEND_DIR, "forecast.html")
    if not os.path.exists(path):
        raise HTTPException(404, "frontend/forecast.html not found")
    return FileResponse(path)


if os.path.isdir(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
