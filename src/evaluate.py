import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, mean_absolute_error, brier_score_loss

class LPLModelEvaluator:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models = {}
        self.load_models()
        
    def load_models(self):
        model_files = [
            "xgb_winner.pkl", "lr_winner.pkl", "rf_score.pkl", "poisson_wickets.pkl",
            "feature_names_win.pkl", "feature_names_score.pkl", "feature_names_wickets.pkl"
        ]
        for f_name in model_files:
            path = os.path.join(self.models_dir, f_name)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.models[f_name] = pickle.load(f)
            else:
                print(f"Warning: {f_name} not found in {self.models_dir}.")

    def evaluate_all(self, processed_df):
        """
        Loads the saved models, extracts the test set, and performs standard evaluation metrics.
        """
        # Feature names
        feat_win = self.models.get("feature_names_win.pkl")
        feat_score = self.models.get("feature_names_score.pkl")
        feat_wickets = self.models.get("feature_names_wickets.pkl")
        
        if not all([feat_win, feat_score, feat_wickets]):
            raise ValueError("Feature names are missing from the models directory.")
            
        # 1. Match Winner Evaluation
        X_win = processed_df[feat_win]
        y_win = processed_df['team1_won'].astype(int)
        
        # Split using the same random state to keep test set consistency
        from sklearn.model_selection import train_test_split
        _, X_test_w, _, y_test_w = train_test_split(X_win, y_win, test_size=0.2, random_state=42)
        
        xgb_winner = self.models.get("xgb_winner.pkl")
        lr_winner = self.models.get("lr_winner.pkl")
        
        print("\n==================================================")
        print("          LPL 2026 Model Evaluation Results        ")
        print("==================================================")
        
        if xgb_winner:
            y_pred_xgb = xgb_winner.predict(X_test_w)
            y_prob_xgb = xgb_winner.predict_proba(X_test_w)[:, 1]
            
            acc = accuracy_score(y_test_w, y_pred_xgb)
            loss = log_loss(y_test_w, y_prob_xgb)
            brier = brier_score_loss(y_test_w, y_prob_xgb)
            
            print("--- Match Winner (XGBoost) ---")
            print(f"  Accuracy:         {acc * 100:.2f}%")
            print(f"  Log-Loss:         {loss:.4f}")
            print(f"  Brier Score:      {brier:.4f}")
            
            # Feature importance
            print("\n  Top Feature Importances (XGBoost):")
            importances = xgb_winner.feature_importances_
            feat_imp = sorted(zip(feat_win, importances), key=lambda x: x[1], reverse=True)[:5]
            for f, imp in feat_imp:
                print(f"    - {f}: {imp:.4f}")
                
        if lr_winner:
            y_pred_lr = lr_winner.predict(X_test_w)
            y_prob_lr = lr_winner.predict_proba(X_test_w)[:, 1]
            
            acc_lr = accuracy_score(y_test_w, y_pred_lr)
            loss_lr = log_loss(y_test_w, y_prob_lr)
            
            print("\n--- Match Winner (Logistic Regression) ---")
            print(f"  Accuracy:         {acc_lr * 100:.2f}%")
            print(f"  Log-Loss:         {loss_lr:.4f}")
            
        # 2. Score Regressor Evaluation
        X_score = processed_df[feat_score]
        y_score = processed_df['team1_score'].astype(float)
        _, X_test_s, _, y_test_s = train_test_split(X_score, y_score, test_size=0.2, random_state=42)
        
        rf_score = self.models.get("rf_score.pkl")
        if rf_score:
            y_pred_s = rf_score.predict(X_test_s)
            rmse = np.sqrt(mean_squared_error(y_test_s, y_pred_s))
            mae = mean_absolute_error(y_test_s, y_pred_s)
            print("\n--- First Innings Score Regressor (Random Forest) ---")
            print(f"  RMSE (Error StdDev): {rmse:.2f} runs")
            print(f"  MAE (Avg Abs Error):  {mae:.2f} runs")
            
        # 3. Wickets Regressor Evaluation
        X_wickets = processed_df[feat_wickets]
        y_wickets = processed_df['team1_wickets'].astype(float)
        _, X_test_k, _, y_test_k = train_test_split(X_wickets, y_wickets, test_size=0.2, random_state=42)
        
        poisson_wickets = self.models.get("poisson_wickets.pkl")
        if poisson_wickets:
            y_pred_k = poisson_wickets.predict(X_test_k)
            rmse_k = np.sqrt(mean_squared_error(y_test_k, y_pred_k))
            mae_k = mean_absolute_error(y_test_k, y_pred_k)
            print("\n--- Wickets per Innings Regressor (Poisson) ---")
            print(f"  RMSE: {rmse_k:.2f} wickets")
            print(f"  MAE:  {mae_k:.2f} wickets")
        print("==================================================\n")

if __name__ == "__main__":
    from loader import LPLDataLoader
    from preprocess import LPLFeatureEngineer
    
    loader = LPLDataLoader()
    datasets = loader.load_all_data()
    
    fe = LPLFeatureEngineer(datasets)
    processed_df = fe.prepare_match_features()
    
    evaluator = LPLModelEvaluator()
    evaluator.evaluate_all(processed_df)
