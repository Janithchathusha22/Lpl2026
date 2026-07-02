import os
import pandas as pd

class LPLDataLoader:
    def __init__(self, data_dir="."):
        self.data_dir = data_dir
        
    def load_csv(self, filename):
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            # Try to see if there's any file matching with spaces or slight variations
            base, ext = os.path.splitext(filename)
            matched_files = [f for f in os.listdir(self.data_dir) if f.startswith(base) and f.endswith(ext)]
            if matched_files:
                path = os.path.join(self.data_dir, matched_files[0])
            else:
                raise FileNotFoundError(f"Required data file not found: {filename} or variations in {self.data_dir}")
        
        try:
            df = pd.read_csv(path)
            # Basic cleaning: strip column names and string values of leading/trailing spaces
            df.columns = [c.strip() for c in df.columns]
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.strip()
            return df
        except Exception as e:
            raise IOError(f"Error reading {path}: {str(e)}")

    def load_all_data(self):
        """
        Loads all essential datasets for the LPL 2026 Prediction system.
        """
        datasets = {}
        
        # Mapping standard internal keys to their filenames
        files_to_load = {
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
        }
        
        for key, filename in files_to_load.items():
            try:
                datasets[key] = self.load_csv(filename)
                # Check for empty dataframe
                if datasets[key].empty:
                    print(f"Warning: {filename} loaded but is empty.")
            except FileNotFoundError as e:
                print(f"Warning: {str(e)}. Proceeding without it.")
                datasets[key] = None
        
        return datasets

if __name__ == "__main__":
    loader = LPLDataLoader()
    data = loader.load_all_data()
    print("Successfully verified loading datasets:")
    for k, v in data.items():
        if v is not None:
            print(f" - {k}: {v.shape[0]} rows, {v.shape[1]} columns")
        else:
            print(f" - {k}: FAILED TO LOAD")
