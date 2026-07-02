import numpy as np
import pandas as pd

def calculate_matchup_advantage(matchup_sr, dismissal_count, balls_faced, league_avg_sr=135.0):
    """
    Calculates Head-to-Head Matchup Advantage Score.
    
    Formula:
      Dismissal Freq % = (Dismissals / Balls Faced) * 100
      SR Index = (Matchup SR / League Average SR) * 100
      Advantage Score = SR Index - (Dismissal Freq % * 10)
      
    Interpretation:
      >= 100: Batsman Advantage
      80-99: Even Matchup
      < 80: Bowler Advantage
    """
    if balls_faced <= 0:
        return 100.0  # Default even/neutral state if no history
    
    dismissal_freq = (dismissal_count / balls_faced) * 100.0
    sr_index = (matchup_sr / league_avg_sr) * 100.0
    advantage_score = sr_index - (dismissal_freq * 10.0)
    return float(advantage_score)

def calculate_win_probability(runs_scored, balls_bowled, target, wickets_lost, chasing=True):
    """
    Calculates current Win Probability (WP) % based on a logistic model.
    
    Formula:
      RRR (Required Run Rate) = (Runs Required / Balls Remaining) * 6
      CRR (Current Run Rate) = (Runs Scored / Balls Bowled) * 6
      Wickets in Hand = 10 - Wickets Lost
      z = (CRR - RRR) * 0.35 + (Wickets in Hand - 5) * 0.45
      Win Prob = 100 / (1 + e^-z)
    """
    balls_remaining = 120 - balls_bowled
    runs_required = target - runs_scored
    
    # Boundary cases
    if runs_required <= 0:
        return 100.0 if chasing else 0.0
    if balls_remaining <= 0:
        return 0.0 if chasing else 100.0
    if wickets_lost >= 10:
        return 0.0 if chasing else 100.0
        
    # Calculate rates
    crr = (runs_scored / max(balls_bowled, 1)) * 6.0
    rrr = (runs_required / balls_remaining) * 6.0
    wickets_in_hand = 10 - wickets_lost
    
    z = (crr - rrr) * 0.35 + (wickets_in_hand - 5) * 0.45
    # Cap z to avoid overflow in exp
    z = np.clip(z, -20.0, 20.0)
    
    wp_chasing = 100.0 / (1.0 + np.exp(-z))
    return wp_chasing if chasing else (100.0 - wp_chasing)

def calculate_pressure_index(runs_scored, balls_bowled, target, wickets_lost):
    """
    Calculates current Pressure Index (scale 0-100).
    
    Formula:
      Pressure Ratio = RRR / max(CRR, 4)
      Wickets Factor = (10 - Wickets Lost) / 10
      Pressure Index = (Pressure Ratio * 50) / Wickets Factor
      Clamped between 0 and 100.
    """
    balls_remaining = 120 - balls_bowled
    runs_required = target - runs_scored
    
    if runs_required <= 0 or wickets_lost >= 10:
        return 100.0 if wickets_lost >= 10 else 0.0
    if balls_remaining <= 0:
        return 100.0
        
    crr = (runs_scored / max(balls_bowled, 1)) * 6.0
    rrr = (runs_required / balls_remaining) * 6.0
    
    pressure_ratio = rrr / max(crr, 4.0)
    wickets_factor = (10.0 - wickets_lost) / 10.0
    
    # Avoid division by zero if wickets_factor is somehow 0
    if wickets_factor <= 0:
        return 100.0
        
    pressure_index = (pressure_ratio * 50.0) / wickets_factor
    return float(np.clip(pressure_index, 0.0, 100.0))

class LPLFeatureEngineer:
    def __init__(self, datasets):
        self.datasets = datasets
        
    def prepare_match_features(self):
        """
        Processes the raw match simulation dataset to create model-ready features.
        """
        df = self.datasets.get("match_simulations")
        if df is None:
            raise ValueError("Match simulations dataset is required for feature engineering.")
            
        processed_df = df.copy()
        
        # 1. Team rating differences
        processed_df['bat_rating_diff'] = processed_df['team1_bat_rating'] - processed_df['team2_bat_rating']
        processed_df['bowl_rating_diff'] = processed_df['team1_bowl_rating'] - processed_df['team2_bowl_rating']
        processed_df['exp_rating_diff'] = processed_df['team1_exp_rating'] - processed_df['team2_exp_rating']
        
        # 2. Home advantage feature
        processed_df['home_advantage_diff'] = processed_df['team1_is_home'].astype(int) - processed_df['team2_is_home'].astype(int)
        
        # 3. Target values
        # team1_won is our binary classification target (1 if team1 won, 0 if team2 won)
        # We also have team1_score, team2_score, team1_wickets, team2_wickets
        
        return processed_df

    def get_team_squad_ratings(self):
        """
        Dynamically aggregates player ratings to calculate team overall attributes.
        """
        df_players = self.datasets.get("player_master")
        df_skills = self.datasets.get("player_skill_ratings")
        
        if df_players is None or df_skills is None:
            return None
            
        # Merge player roles with ratings
        df_merged = pd.merge(df_players, df_skills, on=['player_id', 'player_name', 'team', 'category'])
        
        team_ratings = []
        for team, group in df_merged.groupby('team'):
            # Top 3 batters by powerplay/middle overs ratings
            top_batters = group.nlargest(3, 'batting_middle_over_rating')
            avg_top_batting = top_batters[['batting_power_play_rating', 'batting_middle_over_rating', 'batting_death_over_rating']].mean().mean()
            
            # Top 4 bowlers by powerplay/death overs ratings
            top_bowlers = group.nlargest(4, 'bowling_death_over_rating')
            avg_top_bowling = top_bowlers[['bowling_power_play_rating', 'bowling_middle_over_rating', 'bowling_death_over_rating']].mean().mean()
            
            # Count of genuine all-rounders (role has All-Rounder, skill > 5 in both bat & bowl)
            all_rounders = group[(group['role'].str.contains('All-Rounder', case=False, na=False)) & 
                                 (group['batting_middle_over_rating'] >= 5) & 
                                 (group['bowling_middle_over_rating'] >= 5)]
            allrounder_depth = len(all_rounders)
            
            # General features
            avg_exp = group['intl_experience'].map({'High': 9.0, 'Medium': 7.0, 'Low': 5.0}).fillna(6.0).mean()
            avg_pressure = group['pressure_handling'].mean()
            avg_consistency = group['consistency_score'].mean()
            
            team_ratings.append({
                "team": team,
                "batting_rating": round(avg_top_batting, 2),
                "bowling_rating": round(avg_top_bowling, 2),
                "allrounder_depth": allrounder_depth,
                "experience_rating": round(avg_exp, 2),
                "pressure_handling": round(avg_pressure, 2),
                "consistency": round(avg_consistency, 2)
            })
            
        return pd.DataFrame(team_ratings)

if __name__ == "__main__":
    from loader import LPLDataLoader
    loader = LPLDataLoader()
    datasets = loader.load_all_data()
    
    print("Testing Preprocessing Functions:")
    print("Matchup Adv (Kusal Mendis vs Hasaranga):", calculate_matchup_advantage(128.6, 0, 21))
    print("Win Prob Chasing (91 runs, 12 overs gone (72 balls bowled), target 162, 4 wickets):", 
          calculate_win_probability(91, 72, 162, 4, chasing=True))
    print("Pressure Index:", calculate_pressure_index(91, 72, 162, 4))
    
    fe = LPLFeatureEngineer(datasets)
    match_feats = fe.prepare_match_features()
    print("Feature Engineered Match shape:", match_feats.shape)
    
    squad_feats = fe.get_team_squad_ratings()
    if squad_feats is not None:
        print("\nAggregated Squad Ratings:")
        print(squad_feats)
