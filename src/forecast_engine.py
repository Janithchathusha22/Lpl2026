"""
LPL 2026 — Tournament Forecast Engine
=====================================
Reads the fixture schedule, predicts every group match using the trained
match model, runs a Monte-Carlo tournament simulation (who reaches the
play-offs / final / lifts the cup), and generates an AI tactical game plan
per match from ground + time + dew conditions.

This module is deliberately decoupled from FastAPI / PyTorch. The backend
wires in two callbacks:
    predict_fn(team1, team2, venue, team1_home, dew, match_timing) -> win_prob_team1 (0-100)
    pitch_fn(venue, match_timing, dew) -> pitch profile dict (or None)
"""

import os
import csv
import random
from datetime import datetime

# ─── Name / venue mapping between schedule labels and model labels ────────────
SCHEDULE_TEAM_MAP = {
    "Jaffna Kings": "Jaffna Kings",
    "Galle Titans": "Galle Gallants",
    "Kandy": "Kandy Royals",
    "Dambulla Thunders": "Dambulla Sixers",
    "Colombo Strikers": "Colombo Kaps",
}

GROUND_VENUE_MAP = {
    "Rangiri Dambulla International Cricket Stadium": "Rangiri Dambulla",
    "Pallekele International Cricket Stadium": "Pallekele Stadium",
    "R. Premadasa International Cricket Stadium": "R. Premadasa Stadium",
    "SSC Ground": "SSC Ground",
}

# Each venue's "home" franchise (city based). Galle / Jaffna have no home ground
# in this edition, so they are always treated as the away side.
VENUE_HOME_TEAM = {
    "Rangiri Dambulla": "Dambulla Sixers",
    "Pallekele Stadium": "Kandy Royals",
    "R. Premadasa Stadium": "Colombo Kaps",
    "SSC Ground": "Colombo Kaps",
}

FINAL_VENUE = "R. Premadasa Stadium"
SCHEDULE_YEAR = 2026


def _map_team(name):
    name = (name or "").strip()
    return SCHEDULE_TEAM_MAP.get(name, name)


def _map_venue(ground):
    ground = (ground or "").strip()
    return GROUND_VENUE_MAP.get(ground, ground)


def _parse_date(date_str):
    for fmt in ("%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(f"{(date_str or '').strip()} {SCHEDULE_YEAR}", fmt)
        except ValueError:
            continue
    return None


def _is_night(time_str):
    """19:30 etc → night (dew). 15:00 etc → afternoon."""
    try:
        hour = int(str(time_str).split(":")[0])
        return hour >= 18
    except (ValueError, AttributeError):
        return False


# ─── Schedule loading ─────────────────────────────────────────────────────────
def load_schedule(path):
    """Read the raw fixture CSV into a list of normalised match dicts."""
    if not os.path.exists(path):
        return []

    matches = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            stage = (raw.get("Stage") or "").strip()
            t1_orig = (raw.get("Team 1") or "").strip()
            t2_orig = (raw.get("Team 2") or "").strip()
            ground = (raw.get("Ground") or "").strip()
            venue = _map_venue(ground)
            time_str = (raw.get("Time") or "").strip()
            night = _is_night(time_str)
            dt = _parse_date(raw.get("Date"))
            home_team = VENUE_HOME_TEAM.get(venue)
            is_decided = t1_orig.upper() != "TBD" and t2_orig.upper() != "TBD"

            matches.append({
                "match_no": (raw.get("Match No") or "").strip(),
                "date": (raw.get("Date") or "").strip(),
                "iso_date": dt.date().isoformat() if dt else None,
                "day_of_week": dt.strftime("%A") if dt else None,
                "time": time_str,
                "stage": stage,
                "ground": ground,
                "city": (raw.get("City") or "").strip(),
                "venue": venue,
                "match_timing": "Night" if night else "Day",
                "dew": night,
                "team1_orig": t1_orig,
                "team2_orig": t2_orig,
                "team1": _map_team(t1_orig) if is_decided else None,
                "team2": _map_team(t2_orig) if is_decided else None,
                "home_team": home_team,
                "is_group": stage.lower().startswith("group"),
                "is_decided": is_decided,
            })
    return matches


# ─── Per-match prediction + tactical plan ────────────────────────────────────
def _tactical_plan(match, pitch):
    """Build an AI tactical game plan from ground / time / dew conditions."""
    pitch = pitch or {}
    night = match["dew"]
    pitch_type = str(pitch.get("pitch_type", "Balanced"))
    spin_adv = float(pitch.get("spin_advantage_pct", 50) or 50)
    pace_adv = float(pitch.get("pace_advantage_pct", 50) or 50)
    bat_first_wr = float(pitch.get("bat_first_win_rate", 50) or 50)
    dew_factor = str(pitch.get("dew_factor", "Low"))
    safe_target = pitch.get("safe_score_target")

    spin_heavy = spin_adv >= pace_adv
    bowl_first = night or bat_first_wr < 50

    if night:
        toss_decision = "Win toss → BOWL FIRST"
        bat_choice = "Chase"
        condition_note = (
            f"Heavy dew expected ({dew_factor}). Ball gets wet & skids on — "
            f"second innings batting is easier, so chasing is the percentage call."
        )
    else:
        toss_decision = "Win toss → BAT FIRST"
        bat_choice = "Defend"
        condition_note = (
            f"Afternoon match — pitch bakes under the sun and slows down in the "
            f"second innings. Post a total and defend."
        )

    if spin_heavy:
        bowling_strategy = (
            f"{pitch_type} surface: spin-heavy plan. Deploy a minimum of 8 overs "
            f"of spin, especially through the middle overs to choke the run rate."
        )
        key_bowler_type = "Spinners"
    else:
        bowling_strategy = (
            f"{pitch_type} surface: pace & bounce on offer. Lead with seamers up "
            f"top and keep a death-overs pace option in reserve."
        )
        key_bowler_type = "Pace bowlers"

    return {
        "toss_decision": toss_decision,
        "bat_or_bowl": bat_choice,
        "condition_note": condition_note,
        "bowling_strategy": bowling_strategy,
        "key_bowler_type": key_bowler_type,
        "spin_heavy": spin_heavy,
        "bowl_first": bowl_first,
        "pitch_type": pitch_type,
        "spin_advantage_pct": round(spin_adv, 1),
        "pace_advantage_pct": round(pace_adv, 1),
        "dew_factor": dew_factor,
        "safe_score_target": safe_target,
        "summary": (
            f"{match.get('day_of_week') or ''} {'night' if night else 'afternoon'} game at "
            f"{match.get('city')}. {condition_note} {toss_decision}. {bowling_strategy}"
        ).strip(),
    }


def build_match_predictions(matches, predict_fn, pitch_fn):
    """Attach win probabilities + tactical plan to every decided group match."""
    enriched = []
    for m in matches:
        item = dict(m)
        if m["is_decided"] and m["team1"] and m["team2"]:
            home = m["home_team"]
            team1_home = (home == m["team1"])
            try:
                wp1 = float(predict_fn(
                    m["team1"], m["team2"], m["venue"],
                    team1_home, m["dew"], m["match_timing"],
                ))
            except Exception:
                wp1 = 50.0
            wp1 = max(1.0, min(99.0, wp1))
            item["win_prob_team1"] = round(wp1, 1)
            item["win_prob_team2"] = round(100 - wp1, 1)
            item["predicted_winner"] = m["team1"] if wp1 >= 50 else m["team2"]
            item["home_side"] = home if home in (m["team1"], m["team2"]) else None
        else:
            item["win_prob_team1"] = None
            item["win_prob_team2"] = None
            item["predicted_winner"] = None
            item["home_side"] = None

        try:
            pitch = pitch_fn(m["venue"], m["match_timing"], m["dew"])
        except Exception:
            pitch = None
        item["tactical_plan"] = _tactical_plan(m, pitch)
        enriched.append(item)
    return enriched


# ─── Monte-Carlo tournament simulation ───────────────────────────────────────
def _pairwise_prob(team_a, team_b, predict_fn, cache, venue=FINAL_VENUE):
    """Win prob of team_a vs team_b at the (neutral) final venue, cached."""
    key = (team_a, team_b)
    if key in cache:
        return cache[key]
    home = VENUE_HOME_TEAM.get(venue)
    team1_home = (home == team_a)
    try:
        wp = float(predict_fn(team_a, team_b, venue, team1_home, True, "Night"))
    except Exception:
        wp = 50.0
    wp = max(1.0, min(99.0, wp)) / 100.0
    cache[key] = wp
    cache[(team_b, team_a)] = 1.0 - wp
    return wp


def simulate_tournament(group_matches, teams, predict_fn, n_sims=5000, seed=42):
    """
    Monte-Carlo over the group stage + play-off bracket.
    Returns per-team probabilities of reaching play-offs (top 4),
    the final, and winning the cup, plus expected points.
    """
    rng = random.Random(seed)
    pair_cache = {}

    # Pre-extract group games with their team1-win probability.
    games = [
        (g["team1"], g["team2"], (g["win_prob_team1"] or 50.0) / 100.0)
        for g in group_matches
        if g["is_decided"] and g.get("win_prob_team1") is not None
    ]

    semi = {t: 0 for t in teams}      # reached play-offs (top 4)
    final = {t: 0 for t in teams}     # reached final
    champ = {t: 0 for t in teams}     # won the cup
    points_sum = {t: 0.0 for t in teams}

    for _ in range(n_sims):
        pts = {t: 0 for t in teams}
        nrr = {t: 0.0 for t in teams}  # random tiebreak proxy
        for t in teams:
            nrr[t] = rng.random()
        for t1, t2, p1 in games:
            if rng.random() < p1:
                pts[t1] += 2
            else:
                pts[t2] += 2
        for t in teams:
            points_sum[t] += pts[t]

        ranked = sorted(teams, key=lambda t: (pts[t], nrr[t]), reverse=True)
        top4 = ranked[:4]
        for t in top4:
            semi[t] += 1

        if len(top4) < 4:
            continue
        s1, s2, s3, s4 = top4

        # Qualifier 1: s1 vs s2  → winner to final, loser to Q2
        q1_w, q1_l = (s1, s2) if rng.random() < _pairwise_prob(s1, s2, predict_fn, pair_cache) else (s2, s1)
        # Eliminator: s3 vs s4   → winner to Q2
        el_w = s3 if rng.random() < _pairwise_prob(s3, s4, predict_fn, pair_cache) else s4
        # Qualifier 2: q1_l vs el_w → winner to final
        q2_w = q1_l if rng.random() < _pairwise_prob(q1_l, el_w, predict_fn, pair_cache) else el_w

        final[q1_w] += 1
        final[q2_w] += 1

        # Final
        cup = q1_w if rng.random() < _pairwise_prob(q1_w, q2_w, predict_fn, pair_cache) else q2_w
        champ[cup] += 1

    def pct(d, t):
        return round(100.0 * d[t] / n_sims, 1) if n_sims else 0.0

    progression = []
    for t in teams:
        progression.append({
            "team": t,
            "semi_pct": pct(semi, t),       # reaches play-offs (top 4)
            "final_pct": pct(final, t),
            "champion_pct": pct(champ, t),
            "expected_points": round(points_sum[t] / n_sims, 2) if n_sims else 0.0,
        })
    progression.sort(key=lambda r: (r["champion_pct"], r["final_pct"], r["semi_pct"]), reverse=True)
    return progression


def project_standings(group_matches, teams):
    """Deterministic expected table: expected wins/points from win probabilities."""
    exp_pts = {t: 0.0 for t in teams}
    exp_wins = {t: 0.0 for t in teams}
    played = {t: 0 for t in teams}
    for g in group_matches:
        if not (g["is_decided"] and g.get("win_prob_team1") is not None):
            continue
        t1, t2 = g["team1"], g["team2"]
        p1 = (g["win_prob_team1"] or 50.0) / 100.0
        exp_wins[t1] += p1
        exp_wins[t2] += (1 - p1)
        exp_pts[t1] += p1 * 2
        exp_pts[t2] += (1 - p1) * 2
        played[t1] += 1
        played[t2] += 1

    table = []
    for t in teams:
        table.append({
            "team": t,
            "matches": played[t],
            "expected_wins": round(exp_wins[t], 1),
            "expected_losses": round(played[t] - exp_wins[t], 1),
            "expected_points": round(exp_pts[t], 1),
        })
    table.sort(key=lambda r: r["expected_points"], reverse=True)
    for i, row in enumerate(table, 1):
        row["projected_position"] = i
        row["qualifies"] = i <= 4
    return table


def build_forecast(schedule_path, predict_fn, pitch_fn, n_sims=5000):
    """Top-level orchestration → full forecast payload."""
    raw = load_schedule(schedule_path)
    matches = build_match_predictions(raw, predict_fn, pitch_fn)
    group_matches = [m for m in matches if m["is_group"]]

    teams = sorted({m["team1"] for m in group_matches if m["team1"]} |
                   {m["team2"] for m in group_matches if m["team2"]})

    standings = project_standings(group_matches, teams)
    progression = simulate_tournament(group_matches, teams, predict_fn, n_sims=n_sims)

    # Most-likely play-off projection (fill the TBD bracket from expected seeds).
    seeds = [r["team"] for r in standings[:4]]
    champion = progression[0]["team"] if progression else None
    runner_up = progression[1]["team"] if len(progression) > 1 else None

    return {
        "teams": teams,
        "n_simulations": n_sims,
        "matches": matches,
        "group_matches": group_matches,
        "standings_projection": standings,
        "progression": progression,
        "playoff_seeds": seeds,
        "projected_champion": champion,
        "projected_runner_up": runner_up,
    }
