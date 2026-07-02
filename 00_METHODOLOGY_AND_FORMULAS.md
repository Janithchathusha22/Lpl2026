# LPL 2026 — Advanced Analytics Methodology

Mehema document eke thiyenne, ape 3 CSV files (Head-to-Head Matchup, Win Probability Added, Pressure Index) hadanna use kala **formulas, algorithms, saha statistical methods**. Step-by-step calculate karana widiya saha ML model train karanna recommend karana method eka pahala denawa.

---

## 1️⃣ Head-to-Head Matchup Advantage Score
**File:** `01_head_to_head_matchup_advantage.csv`

### Mokakda meka?
Specific batsman ekak, specific bowler ekata erehiwa kohomada perform karanawada kiyana matchup-level metric eka. Eka general batting/bowling rating ekata wada specific.

### Formula

```
Strike Rate (Matchup)        = (Runs Scored ÷ Balls Faced) × 100

Dismissal Frequency %        = (Dismissals ÷ Balls Faced) × 100

SR Index                     = (Matchup SR ÷ League Average SR) × 100
                                [League Average SR ≈ 135 for T20]

Matchup Advantage Score      = SR Index − (Dismissal Frequency % × 10)
```

### Interpretation
| Score Range | Meaning |
|---|---|
| ≥ 100 | Batsman ට advantage — bowler ekata weak point ekak |
| 80 – 99 | Even matchup |
| < 80 | Bowler ට advantage — batsman ට weakness ekak |

### Worked Example
Kusal Mendis vs Wanindu Hasaranga: 28 balls faced, 38 runs, 1 dismissal.
```
SR = (38 ÷ 28) × 100 = 135.7
Dismissal Freq = (1 ÷ 28) × 100 = 3.57%
SR Index = (135.7 ÷ 135) × 100 = 100.5
Advantage Score = 100.5 − (3.57 × 10) = 64.8  → Bowler ට Advantage
```
(Wicket ekak gihilla thiyena nisa, raw SR eka high unath, score eka adu wenawa — meka real-world ehema wenawa, wicket ekak gihilla thiyenawanam danger ekak)

### Model Train Karanna Recommend Karana Method
- **Weighted Scoring Model** (above formula eka itself use karanna puluwan — simple, explainable)
- Data ginanak thiyenawanam → **Logistic Regression** (dismissal probability predict karanna) + **Linear Regression** (expected SR predict karanna) dekama combine karala "Matchup Win Index" ekak hadanna puluwan
- **Bayesian Shrinkage**: balls faced ginana adu unama (eg 10ක් witharak), sample size eka adui nisa estimate eka league average ekata "shrink" karanna one — ehemath nattam small-sample matchups misleading wenawa

---

## 2️⃣ Win Probability Added (WPA)
**File:** `02_win_probability_added_wpa.csv`

### Mokakda meka?
Cricket analytics walata most powerful metric eka. Ball ekak/over ekak gihilla, **win probability eka** kohomada wenas wenawada kiyala measure karanawa. ESPNcricinfo eke "Forecaster" eka, CrickViz eke models okkoma meka use karanawa.

### Step 1 — Win Probability Formula (Logistic Function)
```
RRR (Required Run Rate) = (Runs Required ÷ Balls Remaining) × 6
CRR (Current Run Rate)  = (Runs Scored ÷ Balls Bowled) × 6
Wickets in Hand          = 10 − Wickets Lost

z = (CRR − RRR) × 0.35 + (Wickets in Hand − 5) × 0.45

Win Probability (Chasing Team) % = 100 ÷ (1 + e^(−z))
```
Meka **Logistic Function** ekak (Sigmoid) — z eka positive unama WP eka 50%ට wadi wenawa, negative unama adu wenawa. Mehema coefficients (0.35, 0.45) anubhavayen / historical data walin calibrate karanna one — dann thiyenne approximate starting values.

### Step 2 — WPA Calculate Karana Widiya
```
WPA(over) = WP(after over) − WP(before over)
```
Over ekak gihilla WP eka 45% → 58% wenawanam, WPA = +13% (chasing team ekata hodai)

### Step 3 — Player Attribution
Over ekak athule ball godak thiyena nisa, WPA eka **proportional attribution** widihata player ekata denawa:
```
Player's WPA Share = Over's Total WPA × (Player's Runs/Wickets in that Over ÷ Total Runs/Wickets in Over)
```

### Worked Example
Over 14: Score 130 → 142 (12 runs), Wickets 3 → 3 (no wicket), Target 172.
```
Balls bowled before = 78, balls remaining before = 42
RRR_before = (172-130)/42 × 6 = 6.0
CRR_before = 130/78 × 6 = 10.0
z_before = (10.0-6.0)×0.35 + (7-5)×0.45 = 1.4+0.9 = 2.3
WP_before = 100/(1+e^-2.3) = 90.9%

After over: Score=142, balls=84, balls remaining=36
RRR_after = (172-142)/36 × 6 = 5.0
CRR_after = 142/84 × 6 = 10.14
z_after = (10.14-5.0)×0.35 + 2×0.45 = 1.8+0.9 = 2.7
WP_after = 100/(1+e^-2.7) = 93.7%

WPA = 93.7 - 90.9 = +2.8%
```

### Model Train Karanna Recommend Karana Method
- **Logistic Regression**: Historical ball-by-ball data (score, wickets, overs, target) → actual match result (win/loss) walin train karala, real coefficients (mehe 0.35/0.45 wagema) learn ganna puluwan
- **Gradient Boosted Trees (XGBoost/LightGBM)**: More accurate — non-linear interactions (eg venue + wickets + dew factor) capture karanna
- Validate karanna: **Log-Loss** ho **Brier Score** use karala model eke calibration eka check karanna (predicted 70% win una match godayak actually 70%ක win wela thiyenawada kiyala)

---

## 3️⃣ Pressure Index (Required Run Rate Based)
**File:** `03_pressure_index_reference_grid.csv`

### Mokakda meka?
Match situation ekaka "pressure level" eka — single number ekakin summarize karana index eka. Batsman kenek dismissal probability eka, pressure wadi welawe wadi wenawa kiyana hypothesis eka base karagena.

### Formula
```
Pressure Ratio   = RRR ÷ max(CRR, 4)        [CRR floor 4ට, divide-by-zero error wenna epa]

Wickets Factor   = (10 − Wickets Lost) ÷ 10  [1.0 = full wickets, 0.1 = 9 down]

Pressure Index   = (Pressure Ratio × 50) ÷ Wickets Factor
                   [clamp 0-100 athule]
```

### Interpretation Scale
| Pressure Index | Level |
|---|---|
| 0 – 34 | Low — comfortable chase |
| 35 – 59 | Moderate — steady batting needed |
| 60 – 79 | High — boundary-dependent |
| 80 – 100 | Extreme — likely defeat zone |

### Worked Example
Overs remaining = 3 (18 balls), RRR = 15, CRR = 9.5, Wickets lost = 6
```
Pressure Ratio = 15 ÷ 9.5 = 1.58
Wickets Factor = (10-6)/10 = 0.4
Pressure Index = (1.58 × 50) ÷ 0.4 = 197.5 → clamp to 100
→ EXTREME PRESSURE
```

### Model Train Karanna Recommend Karana Method
- **Poisson Regression**: Pressure Index eka feature ekak widihata gaththata, "wickets in next over" predict karanna — wickets eka count data nisa Poisson distribution eka hodai
- **Survival Analysis (Cox Proportional Hazards)**: "Batsman ekak kochchara welawak dismiss wenna ne wenawada" kiyana time-to-event modeling — pressure index eka covariate ekak widihata
- Simple use case ekata: Pressure Index eka **direct feature** widihatama Match Winner classifier ekata (XGBoost) add karanna puluwan — model eke accuracy godak wadi wenawa

---

## 📊 Summary — Which Stat/Algorithm for Which Goal

| Goal | Best Method | Why |
|---|---|---|
| Match Winner Prediction | XGBoost / Random Forest | Non-linear feature interactions handle karanawa (venue × team × pressure) |
| Win Probability (live) | Logistic Regression → calibrated with Platt Scaling | Probability output ekak directly denawa, interpretable |
| Player Matchup Advantage | Weighted Scoring + Bayesian Shrinkage | Small sample sizes walata robust |
| Score Prediction (runs) | Linear Regression / Gradient Boosting Regressor | Continuous numeric output |
| Wickets in Innings | Poisson Regression | Count data (wickets 0-10) ekata natural fit |
| Player of Tournament | Weighted Composite Score (manual) → later Neural Net | Multiple stats combine karanna one (bat+bowl+field+pressure) |

### General Statistical Concepts Use Karapu Eka
- **Sigmoid/Logistic Function**: `1/(1+e^-z)` — eka any real number eka 0-1 (probability) ekata convert karanawa
- **Z-Score Normalization**: player rating eka league average ekata compare karaddi use karanawa `(value - mean)/std_dev`
- **Bayesian Shrinkage**: sample size adui welawe, estimate eka population average ekata "pull" karanawa — overfitting wenna epa kiyala
- **Exponentially Weighted Moving Average (EWMA)**: "recent form" calculate karanna — pasugiya match ekata wedi weight ekak, parana match ekata adu weight ekak denawa: `EWMA_t = α × value_t + (1-α) × EWMA_(t-1)`

---
*Mehema formulas okkoma "starting point" widihata hadapu eka — real historical ball-by-ball data thiyenawanam, coefficients (0.35, 0.45 wagema) walata regression run karala accurate values ganna one.*
