"""
AI Tactical Engine for the LPL 2026 delivery simulator.

The current repo does not contain ball-by-ball delivery outcomes. This module
therefore builds a derived tactical training set from existing batting,
bowling, and venue CSVs, trains a classifier, and combines the classifier output
with a deterministic field optimizer.
"""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
TACTICAL_MODEL_FILE = "tactical_engine.pkl"
TACTICAL_REPORT_FILE = "tactical_engine_report.json"

DELIVERY_OPTIONS = [
    {"delivery_type": "Outswinger", "bowling_family": "Pace", "speed_kph": 144, "line": "4th Stump", "length": "Good Length", "movement": "Swing Away", "phase": "Powerplay"},
    {"delivery_type": "Inswinger", "bowling_family": "Pace", "speed_kph": 140, "line": "Middle", "length": "Good Length", "movement": "Swing In", "phase": "Powerplay"},
    {"delivery_type": "Hard Length", "bowling_family": "Pace", "speed_kph": 138, "line": "Body", "length": "Back of Length", "movement": "Seam", "phase": "Middle"},
    {"delivery_type": "Bouncer", "bowling_family": "Pace", "speed_kph": 146, "line": "Body", "length": "Short", "movement": "Bounce", "phase": "Powerplay"},
    {"delivery_type": "Yorker", "bowling_family": "Pace", "speed_kph": 142, "line": "Middle", "length": "Yorker", "movement": "Straight", "phase": "Death"},
    {"delivery_type": "Slower Ball", "bowling_family": "Pace", "speed_kph": 118, "line": "Off Stump", "length": "Full", "movement": "Pace Off", "phase": "Death"},
    {"delivery_type": "Off Spin", "bowling_family": "Spin", "speed_kph": 88, "line": "Off Stump", "length": "Good Length", "movement": "Turn Away", "phase": "Middle"},
    {"delivery_type": "Leg Spin", "bowling_family": "Spin", "speed_kph": 84, "line": "Middle", "length": "Good Length", "movement": "Turn In", "phase": "Middle"},
    {"delivery_type": "Googly", "bowling_family": "Spin", "speed_kph": 86, "line": "4th Stump", "length": "Full", "movement": "Wrong Un", "phase": "Middle"},
    {"delivery_type": "Carrom Ball", "bowling_family": "Spin", "speed_kph": 90, "line": "Off Stump", "length": "Full", "movement": "Skid", "phase": "Death"},
]

CATEGORICAL_FEATURES = [
    "batter_name",
    "batter_style",
    "batter_hand",
    "bowler_name",
    "bowling_type",
    "bowling_family",
    "delivery_type",
    "line",
    "length",
    "movement",
    "phase",
    "venue",
    "pitch_type",
    "dew_factor",
]

NUMERIC_FEATURES = [
    "speed_kph",
    "powerplay_sr",
    "middle_sr",
    "death_sr",
    "boundary_pct",
    "dot_ball_pct",
    "spin_handling",
    "pace_handling",
    "pressure_avg",
    "bowler_matchup_rating",
    "bowler_wickets_per_match",
    "bowler_economy_phase",
    "pace_advantage_pct",
    "spin_advantage_pct",
    "safe_score_target",
    "powerplay_avg_wickets",
    "death_over_avg_wickets",
    "humidity_pct",
    "temperature_c",
]

OUTCOME_PRIORITY = ["Out - Caught", "Out - Bowled/LBW", "Dot Ball", "1 Run", "4 Runs", "6 Runs"]


def _read_csv(data_dir: str, filename: str) -> pd.DataFrame:
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
    return df


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        parsed = float(value)
        if np.isnan(parsed) or np.isinf(parsed):
            return float(default)
        return parsed
    except Exception:
        return float(default)


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return default
    return text


def _bowling_family(bowling_type: str) -> str:
    text = bowling_type.lower()
    if any(token in text for token in ["spin", "carrom", "googly", "leg", "off", "mystery", "orthodox"]):
        return "Spin"
    return "Pace"


def _batter_hand(row: pd.Series) -> str:
    text = _text(row.get("batting_style"), "").lower()
    return "LHB" if "left" in text or "lhb" in text else "RHB"


def _phase_rating_column(hand: str, phase: str) -> str:
    phase_l = phase.lower()
    if "power" in phase_l:
        suffix = "powerplay"
    elif "death" in phase_l:
        suffix = "death"
    else:
        suffix = "middle"
    return f"vs_{hand}_{suffix}"


def _phase_economy_column(phase: str) -> str:
    phase_l = phase.lower()
    if "power" in phase_l:
        return "economy_powerplay"
    if "death" in phase_l:
        return "economy_death"
    return "economy_middle"


def _matchup_rating(bowler: pd.Series, hand: str, phase: str) -> float:
    return _num(bowler.get(_phase_rating_column(hand, phase)), 6.5)


def _economy_phase(bowler: pd.Series, phase: str) -> float:
    return _num(bowler.get(_phase_economy_column(phase)), 7.5)


def _derive_delivery_label(row: Dict[str, Any]) -> Dict[str, Any]:
    family = row["bowling_family"]
    movement = str(row["movement"]).lower()
    length = str(row["length"]).lower()
    line = str(row["line"]).lower()
    pitch = str(row["pitch_type"]).lower()
    weakness = str(row.get("weakness", "")).lower()
    batter_style = str(row["batter_style"]).lower()

    matchup = _num(row["bowler_matchup_rating"], 6.5)
    boundary = _num(row["boundary_pct"], 42)
    dots = _num(row["dot_ball_pct"], 24)
    spin_handling = _num(row["spin_handling"], 7)
    pace_handling = _num(row["pace_handling"], 7)
    pace_adv = _num(row["pace_advantage_pct"], 50)
    spin_adv = _num(row["spin_advantage_pct"], 50)
    pp_wickets = _num(row["powerplay_avg_wickets"], 1.2)
    death_wickets = _num(row["death_over_avg_wickets"], 2.4)
    dew_high = str(row["dew_factor"]).lower() == "high"

    pace_vuln = max(0.0, 10.0 - pace_handling) + max(0.0, matchup - 6.5) + max(0.0, pace_adv - 50.0) / 12.0
    spin_vuln = max(0.0, 10.0 - spin_handling) + max(0.0, matchup - 6.5) + max(0.0, spin_adv - 50.0) / 12.0
    wicket_pressure = pp_wickets if row["phase"] == "Powerplay" else death_wickets if row["phase"] == "Death" else (pp_wickets + death_wickets) / 2.0

    caught_score = 0.0
    bowled_lbw_score = 0.0
    boundary_score = boundary / 12.0
    dot_score = dots / 7.0 + max(0.0, 8.0 - matchup) / 2.0
    one_score = 3.0

    shot = "Defensive Push"
    target = "Point"
    dismissal = "None"

    if family == "Pace":
        if ("away" in movement or "4th" in line) and ("good" in length or "back" in length):
            caught_score = pace_vuln + wicket_pressure + 3.6
            shot = "Thick Edge"
            target = "1st Slip"
            dismissal = "Caught"
        elif "short" in length or "bounce" in movement or "body" in line:
            caught_score = pace_vuln + 2.4
            boundary_score += 2.0 if boundary >= 48 else 0.8
            shot = "Pull Shot"
            target = "Deep Square"
            dismissal = "Caught"
        elif "yorker" in length or "middle" in line:
            bowled_lbw_score = pace_vuln + wicket_pressure + 2.8
            shot = "Late Bat Down"
            target = "Mid On"
            dismissal = "Bowled/LBW"
        elif "pace off" in movement:
            caught_score = pace_vuln + 2.1
            dot_score += 1.4
            shot = "Mistimed Loft"
            target = "Long Off"
            dismissal = "Caught"
    else:
        if spin_adv >= 58 or any(token in pitch for token in ["spin", "turn", "dry", "dust"]):
            caught_score = spin_vuln + wicket_pressure + 3.2
        if "wrong" in movement or "skid" in movement:
            bowled_lbw_score = spin_vuln + 2.5
            shot = "Misread Spin"
            target = "Short Cover"
            dismissal = "Caught"
        elif "off" in line or "4th" in line:
            caught_score += 1.3
            shot = "Lofted Drive"
            target = "Long Off"
            dismissal = "Caught"
        else:
            shot = "Sweep"
            target = "Deep Midwicket"
            dismissal = "Caught"

    if "spin" in weakness and family == "Spin":
        caught_score += 1.8
        bowled_lbw_score += 0.8
    if "pace" in weakness and family == "Pace":
        caught_score += 1.5
        bowled_lbw_score += 0.9
    if any(token in batter_style for token in ["aggressive", "explosive", "hitter"]):
        boundary_score += 1.6
        caught_score += 0.6
    if dew_high and family == "Spin":
        caught_score -= 0.6
        boundary_score += 0.5

    scores = {
        "Out - Caught": caught_score,
        "Out - Bowled/LBW": bowled_lbw_score,
        "Dot Ball": dot_score,
        "1 Run": one_score,
        "4 Runs": boundary_score,
        "6 Runs": boundary_score - 0.8 + (1.2 if boundary >= 52 and "full" in length else 0.0),
    }
    outcome = max(OUTCOME_PRIORITY, key=lambda item: scores[item])
    if outcome == "Out - Caught":
        dismissal = "Caught"
    elif outcome == "Out - Bowled/LBW":
        dismissal = "Bowled/LBW"
        target = "Bowler"
    else:
        dismissal = "None"
        if outcome in {"4 Runs", "6 Runs"}:
            if family == "Pace" and ("short" in length or "body" in line):
                shot = "Pull Shot"
                target = "Deep Square"
            elif family == "Spin":
                shot = "Lofted Sweep"
                target = "Deep Midwicket"
            else:
                shot = "Cover Drive"
                target = "Long Off"
        elif outcome == "Dot Ball":
            shot = "Checked Drive"
            target = "Cover"
        else:
            shot = "Nudge"
            target = "Midwicket"

    raw_wicket = max(caught_score, bowled_lbw_score)
    catch_probability = int(round(np.clip(38 + raw_wicket * 7.2 - boundary_score * 2.2, 8, 93)))
    if outcome != "Out - Caught":
        catch_probability = int(round(np.clip(catch_probability * 0.35, 0, 45)))

    return {
        "outcome": outcome,
        "shot_played": shot,
        "dismissal_type": dismissal,
        "catch_target_node": target,
        "catch_probability": catch_probability,
        "label_source": "derived_from_lpl_csv_features",
    }


def build_tactical_training_frame(data_dir: str = ROOT) -> pd.DataFrame:
    batting = _read_csv(data_dir, "07_batting_profiles.csv")
    bowling = _read_csv(data_dir, "06_bowling_matchup_matrix.csv")
    venues = _read_csv(data_dir, "LPL_2026_Advanced_Pitch_Data.csv")

    rows: List[Dict[str, Any]] = []
    for _, batter in batting.iterrows():
        hand = _batter_hand(batter)
        for _, bowler in bowling.iterrows():
            bowler_family = _bowling_family(_text(bowler.get("bowling_type"), "Pace"))
            for _, venue in venues.iterrows():
                for delivery in DELIVERY_OPTIONS:
                    if delivery["bowling_family"] != bowler_family:
                        continue
                    phase = delivery["phase"]
                    row = {
                        "batter_name": _text(batter.get("player_name")),
                        "batter_style": _text(batter.get("batting_style"), "Balanced"),
                        "batter_hand": hand,
                        "bowler_name": _text(bowler.get("bowler_name")),
                        "bowling_type": _text(bowler.get("bowling_type"), "Unknown"),
                        "bowling_family": bowler_family,
                        "delivery_type": delivery["delivery_type"],
                        "line": delivery["line"],
                        "length": delivery["length"],
                        "movement": delivery["movement"],
                        "phase": phase,
                        "speed_kph": delivery["speed_kph"],
                        "venue": _text(venue.get("Ground_Name")),
                        "pitch_type": _text(venue.get("Pitch_Type"), "Balanced"),
                        "dew_factor": _text(venue.get("Dew_Factor"), "Low"),
                        "powerplay_sr": _num(batter.get("powerplay_sr_estimate"), 130),
                        "middle_sr": _num(batter.get("middle_over_sr_estimate"), 130),
                        "death_sr": _num(batter.get("death_over_sr_estimate"), 140),
                        "boundary_pct": _num(batter.get("boundary_pct"), 42),
                        "dot_ball_pct": _num(batter.get("dot_ball_pct"), 24),
                        "spin_handling": _num(batter.get("spin_handling"), 7),
                        "pace_handling": _num(batter.get("pace_handling"), 7),
                        "pressure_avg": _num(batter.get("pressure_avg"), 25),
                        "weakness": _text(batter.get("weakness")),
                        "bowler_matchup_rating": _matchup_rating(bowler, hand, phase),
                        "bowler_wickets_per_match": _num(bowler.get("wickets_per_match_estimate"), 1.4),
                        "bowler_economy_phase": _economy_phase(bowler, phase),
                        "pace_advantage_pct": _num(venue.get("Pace_Advantage_pct"), 50),
                        "spin_advantage_pct": _num(venue.get("Spin_Advantage_pct"), 50),
                        "safe_score_target": _num(venue.get("Safe_Score_Target"), 165),
                        "powerplay_avg_wickets": _num(venue.get("Powerplay_Avg_Wickets"), 1.2),
                        "death_over_avg_wickets": _num(venue.get("Death_Over_Avg_Wickets"), 2.4),
                        "humidity_pct": _num(venue.get("Humidity_pct"), 75),
                        "temperature_c": _num(venue.get("Temperature_C"), 29),
                    }
                    row.update(_derive_delivery_label(row))
                    rows.append(row)

    return pd.DataFrame(rows)


def _make_pipeline() -> Pipeline:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder, CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        n_estimators=260,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("classifier", classifier)])


def train_tactical_engine(data_dir: str = ROOT, models_dir: str = MODELS_DIR) -> Dict[str, Any]:
    os.makedirs(models_dir, exist_ok=True)
    frame = build_tactical_training_frame(data_dir)
    if frame.empty:
        raise ValueError("Tactical training frame is empty")

    X = frame[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = frame["outcome"]
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42, stratify=stratify
    )

    model = _make_pipeline()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    accuracy = float(accuracy_score(y_test, y_pred))

    bundle = {
        "model_type": "RandomForestClassifier",
        "label_source": "derived_from_lpl_csv_features_not_real_ball_by_ball",
        "trained_at": datetime.utcnow().isoformat(),
        "model": model,
        "features": CATEGORICAL_FEATURES + NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "delivery_options": DELIVERY_OPTIONS,
        "outcome_classes": list(model.named_steps["classifier"].classes_),
        "training_rows": int(len(frame)),
        "source_files": [
            "07_batting_profiles.csv",
            "06_bowling_matchup_matrix.csv",
            "LPL_2026_Advanced_Pitch_Data.csv",
        ],
    }

    model_path = os.path.join(models_dir, TACTICAL_MODEL_FILE)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    report_payload = {
        "trained_at": bundle["trained_at"],
        "model_path": model_path,
        "training_rows": int(len(frame)),
        "test_rows": int(len(X_test)),
        "accuracy": accuracy,
        "class_counts": {str(k): int(v) for k, v in y.value_counts().to_dict().items()},
        "classification_report": report,
        "label_source": bundle["label_source"],
        "source_files": bundle["source_files"],
    }
    with open(os.path.join(models_dir, TACTICAL_REPORT_FILE), "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    return report_payload


def load_tactical_engine(models_dir: str = MODELS_DIR) -> Optional[Dict[str, Any]]:
    path = os.path.join(models_dir, TACTICAL_MODEL_FILE)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _find_row(df: Optional[pd.DataFrame], name_col: str, name: str) -> Optional[pd.Series]:
    if df is None or df.empty or not name:
        return None
    lookup = str(name).strip().lower()
    exact = df[df[name_col].astype(str).str.strip().str.lower() == lookup]
    if not exact.empty:
        return exact.iloc[0]
    contains = df[df[name_col].astype(str).str.lower().str.contains(lookup, regex=False)]
    if not contains.empty:
        return contains.iloc[0]
    return None


def _venue_row(venues: pd.DataFrame, venue_name: str) -> pd.Series:
    if venues is None or venues.empty:
        raise ValueError("Venue profile data not available")
    row = _find_row(venues, "Ground_Name", venue_name)
    if row is not None:
        return row
    return venues.iloc[0]


def _choose_default_bowler(bowling: pd.DataFrame, bowling_type: Optional[str] = None) -> pd.Series:
    candidates = bowling.copy()
    if bowling_type:
        family = _bowling_family(bowling_type)
        candidates = candidates[candidates["bowling_type"].astype(str).apply(_bowling_family) == family]
    if candidates.empty:
        candidates = bowling
    candidates = candidates.copy()
    candidates["_rank"] = candidates["wickets_per_match_estimate"].apply(_num) * 8 - candidates["economy_middle"].apply(_num)
    return candidates.sort_values("_rank", ascending=False).iloc[0]


def _feature_row_for_request(
    datasets: Dict[str, pd.DataFrame],
    request: Dict[str, Any],
    delivery: Dict[str, Any],
) -> Tuple[Dict[str, Any], pd.Series, pd.Series, pd.Series]:
    batting = datasets.get("batting_profiles")
    bowling = datasets.get("bowling_matchup_matrix")
    venues = datasets.get("advanced_pitch_data")
    if batting is None or batting.empty or bowling is None or bowling.empty or venues is None or venues.empty:
        raise ValueError("Tactical CSV inputs are not loaded")

    batter = _find_row(batting, "player_name", _text(request.get("target_batsman")))
    if batter is None:
        batter = batting.iloc[0]

    bowler = _find_row(bowling, "bowler_name", _text(request.get("bowler_name")))
    if bowler is None:
        bowler = _choose_default_bowler(bowling, request.get("bowling_type"))

    venue = _venue_row(venues, _text(request.get("venue"), "R. Premadasa Stadium"))
    hand = _text(request.get("batter_hand"), _batter_hand(batter))
    bowler_family = _bowling_family(_text(bowler.get("bowling_type"), request.get("bowling_type") or "Pace"))
    phase = _text(request.get("match_phase"), delivery["phase"])

    row = {
        "batter_name": _text(batter.get("player_name")),
        "batter_style": _text(batter.get("batting_style"), "Balanced"),
        "batter_hand": hand,
        "bowler_name": _text(bowler.get("bowler_name")),
        "bowling_type": _text(bowler.get("bowling_type"), "Unknown"),
        "bowling_family": bowler_family,
        "delivery_type": delivery["delivery_type"],
        "line": delivery["line"],
        "length": delivery["length"],
        "movement": delivery["movement"],
        "phase": phase,
        "speed_kph": delivery["speed_kph"],
        "venue": _text(venue.get("Ground_Name")),
        "pitch_type": _text(venue.get("Pitch_Type"), "Balanced"),
        "dew_factor": _text(venue.get("Dew_Factor"), "Low"),
        "powerplay_sr": _num(batter.get("powerplay_sr_estimate"), 130),
        "middle_sr": _num(batter.get("middle_over_sr_estimate"), 130),
        "death_sr": _num(batter.get("death_over_sr_estimate"), 140),
        "boundary_pct": _num(batter.get("boundary_pct"), 42),
        "dot_ball_pct": _num(batter.get("dot_ball_pct"), 24),
        "spin_handling": _num(batter.get("spin_handling"), 7),
        "pace_handling": _num(batter.get("pace_handling"), 7),
        "pressure_avg": _num(batter.get("pressure_avg"), 25),
        "weakness": _text(batter.get("weakness")),
        "bowler_matchup_rating": _matchup_rating(bowler, hand, phase),
        "bowler_wickets_per_match": _num(bowler.get("wickets_per_match_estimate"), 1.4),
        "bowler_economy_phase": _economy_phase(bowler, phase),
        "pace_advantage_pct": _num(venue.get("Pace_Advantage_pct"), 50),
        "spin_advantage_pct": _num(venue.get("Spin_Advantage_pct"), 50),
        "safe_score_target": _num(venue.get("Safe_Score_Target"), 165),
        "powerplay_avg_wickets": _num(venue.get("Powerplay_Avg_Wickets"), 1.2),
        "death_over_avg_wickets": _num(venue.get("Death_Over_Avg_Wickets"), 2.4),
        "humidity_pct": _num(venue.get("Humidity_pct"), 75),
        "temperature_c": _num(venue.get("Temperature_C"), 29),
    }
    return row, batter, bowler, venue


def _mirror_x(x: float, hand: str) -> float:
    return -x if hand == "LHB" else x


def _field_setup_for_prediction(predicted: Dict[str, Any], hand: str) -> List[Dict[str, Any]]:
    target = predicted.get("catch_target_node", "")
    outcome = predicted.get("outcome", "")
    shot = predicted.get("shot_played", "")

    if target in {"1st Slip", "Gully"} or "Edge" in shot:
        raw = [
            ("WK", "wk", 0.04, 0.24),
            ("1st Slip", "close", 0.22, 0.38),
            ("2nd Slip", "close", 0.32, 0.36),
            ("Gully", "close", 0.50, 0.24),
            ("Point", "ring", 0.72, 0.02),
            ("Cover", "ring", 0.58, -0.34),
            ("Mid Off", "ring", 0.25, -0.72),
            ("Mid On", "ring", -0.25, -0.72),
            ("Fine Leg", "deep", -0.26, 0.90),
            ("3rd Man", "deep", 0.32, 0.90),
            ("Deep Point", "deep", 0.88, 0.08),
        ]
    elif target in {"Deep Square", "Deep Midwicket"} or "Pull" in shot or "Sweep" in shot:
        raw = [
            ("WK", "wk", 0.04, 0.24),
            ("Slip", "close", 0.20, 0.38),
            ("Point", "ring", 0.68, 0.04),
            ("Cover", "ring", 0.54, -0.36),
            ("Mid Off", "ring", 0.24, -0.70),
            ("Mid On", "ring", -0.20, -0.70),
            ("Midwicket", "ring", -0.55, -0.25),
            ("Square Leg", "ring", -0.68, 0.08),
            ("Deep Square", "deep", -0.86, 0.30),
            ("Deep Midwicket", "deep", -0.72, 0.48),
            ("Long On", "deep", -0.30, -0.90),
        ]
    elif "Bowled" in outcome or "LBW" in outcome:
        raw = [
            ("WK", "wk", 0.04, 0.24),
            ("Short Midwicket", "ring", -0.42, -0.18),
            ("Short Cover", "ring", 0.42, -0.24),
            ("Point", "ring", 0.68, 0.02),
            ("Cover", "ring", 0.54, -0.36),
            ("Mid Off", "ring", 0.24, -0.72),
            ("Mid On", "ring", -0.24, -0.72),
            ("Square Leg", "ring", -0.68, 0.08),
            ("Fine Leg", "deep", -0.24, 0.90),
            ("3rd Man", "deep", 0.30, 0.90),
            ("Long Off", "deep", 0.46, -0.90),
        ]
    else:
        raw = [
            ("WK", "wk", 0.04, 0.24),
            ("Slip", "close", 0.20, 0.38),
            ("Point", "ring", 0.72, 0.02),
            ("Cover", "ring", 0.58, -0.38),
            ("Mid Off", "ring", 0.28, -0.73),
            ("Mid On", "ring", -0.26, -0.73),
            ("Midwicket", "ring", -0.58, -0.32),
            ("Square Leg", "ring", -0.72, 0.05),
            ("Fine Leg", "deep", -0.26, 0.90),
            ("3rd Man", "deep", 0.30, 0.90),
            ("Long Off", "deep", 0.53, -0.88),
        ]

    return [
        {"name": name, "type": role, "x": round(_mirror_x(x, hand), 3), "z": round(z, 3)}
        for name, role, x, z in raw
    ]


def _top_probabilities(model: Pipeline, frame: pd.DataFrame) -> List[Dict[str, Any]]:
    probabilities = model.predict_proba(frame)[0]
    classes = list(model.named_steps["classifier"].classes_)
    pairs = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
    return [{"outcome": str(label), "probability": round(float(prob) * 100, 1)} for label, prob in pairs[:5]]


def predict_tactical_plan(
    bundle: Dict[str, Any],
    datasets: Dict[str, pd.DataFrame],
    request: Dict[str, Any],
) -> Dict[str, Any]:
    if not bundle or "model" not in bundle:
        raise ValueError("Tactical engine model is not loaded")
    model: Pipeline = bundle["model"]

    candidates = []
    for delivery in bundle.get("delivery_options", DELIVERY_OPTIONS):
        row, batter, bowler, venue = _feature_row_for_request(datasets, request, delivery)
        if row["bowling_family"] != _bowling_family(row["bowling_type"]):
            continue
        frame = pd.DataFrame([row])[bundle["features"]]
        probabilities = model.predict_proba(frame)[0]
        classes = list(model.named_steps["classifier"].classes_)
        prob_map = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
        caught_prob = prob_map.get("Out - Caught", 0.0)
        bowled_prob = prob_map.get("Out - Bowled/LBW", 0.0)
        dot_prob = prob_map.get("Dot Ball", 0.0)
        boundary_prob = prob_map.get("4 Runs", 0.0) + prob_map.get("6 Runs", 0.0)
        utility = caught_prob * 1.35 + bowled_prob * 1.2 + dot_prob * 0.25 - boundary_prob * 0.55
        predicted = _derive_delivery_label(row)
        predicted["model_top_probabilities"] = _top_probabilities(model, frame)
        predicted["model_outcome"] = str(model.predict(frame)[0])
        candidates.append((utility, row, predicted, batter, bowler, venue, prob_map))

    if not candidates:
        raise ValueError("No tactical delivery candidates could be generated")

    utility, row, predicted, batter, bowler, venue, prob_map = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
    hand = row["batter_hand"]
    optimal_field = _field_setup_for_prediction(predicted, hand)
    caught_prob = prob_map.get("Out - Caught", 0.0)
    bowled_prob = prob_map.get("Out - Bowled/LBW", 0.0)
    if predicted["dismissal_type"] == "Caught":
        dismissal_probability = int(round(predicted["catch_probability"]))
    elif predicted["dismissal_type"] == "Bowled/LBW":
        dismissal_probability = int(round(np.clip(bowled_prob * 100, 8, 90)))
    else:
        dismissal_probability = int(round(np.clip((caught_prob + bowled_prob) * 100, 0, 45)))

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "model_meta": {
            "model_type": bundle.get("model_type"),
            "trained_at": bundle.get("trained_at"),
            "label_source": bundle.get("label_source"),
            "training_rows": bundle.get("training_rows"),
        },
        "tactical_plan": {
            "target_batsman": row["batter_name"],
            "bowler": row["bowler_name"],
            "venue": row["venue"],
            "recommended_delivery": {
                "type": row["delivery_type"],
                "speed": f"{int(round(row['speed_kph']))} kph",
                "line": row["line"],
                "length": row["length"],
                "movement": row["movement"],
                "phase": row["phase"],
                "bowling_type": row["bowling_type"],
            },
            "expected_outcome": {
                "shot_played": predicted["shot_played"],
                "dismissal_type": predicted["dismissal_type"],
                "outcome": predicted["model_outcome"],
                "probability": dismissal_probability,
                "catch_target_node": predicted["catch_target_node"],
                "top_probabilities": predicted["model_top_probabilities"],
            },
            "optimal_field_setup": optimal_field,
            "analysis": {
                "batter_style": row["batter_style"],
                "batter_hand": row["batter_hand"],
                "batter_weakness": row["weakness"],
                "bowler_matchup_rating": round(float(row["bowler_matchup_rating"]), 2),
                "pitch_type": row["pitch_type"],
                "pace_advantage_pct": row["pace_advantage_pct"],
                "spin_advantage_pct": row["spin_advantage_pct"],
                "utility_score": round(float(utility), 4),
            },
        },
    }


if __name__ == "__main__":
    result = train_tactical_engine(ROOT, MODELS_DIR)
    print(json.dumps(result, indent=2))
