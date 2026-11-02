
import pandas as pd
import numpy as np
import os
import joblib
from scipy.stats import ks_2samp

MODEL_SAVE_PATH = "models/saved_models/"
BASELINE_STATS_PATH = os.path.join(MODEL_SAVE_PATH, "baseline_stats.json")

def detect_drift(new_data_path):
    """
    Checks for data drift by comparing new data against training baseline statistics using KS Test.
    Returns a dictionary of drift results per feature.
    """
    if not os.path.exists(BASELINE_STATS_PATH):
        return {"error": "Baseline statistics not found. Please train model first."}
    
    try:
        baseline_stats = pd.read_json(BASELINE_STATS_PATH)
        
        if not os.path.exists(new_data_path):
             return {"error": f"New data file not found at {new_data_path}"}
             
        new_data = pd.read_csv(new_data_path)
        
        drift_report = {}
        drift_detected = False
        
        # Features to check (all numeric)
        features = [col for col in baseline_stats.columns if col in new_data.columns]
        
        for feature in features:
            # Simple check: compare mean/std first, or do KS test if we had full distribution saved.
            # Since we only saved stats (mean/std/quartiles), we can do a basic threshold check.
            # Example: If new mean is > 2 std devs away from old mean.
            
            old_mean = baseline_stats.loc['mean', feature]
            old_std = baseline_stats.loc['std', feature]
            
            new_mean = new_data[feature].mean()
            
            # Z-score of new mean relative to old distribution
            if old_std > 0:
                z_score = abs(new_mean - old_mean) / old_std
            else:
                z_score = 0 if new_mean == old_mean else 999
                
            is_drift = z_score > 3 # Threshold
            
            if is_drift:
                drift_detected = True
                
            drift_report[feature] = {
                "drift_detected": bool(is_drift),
                "z_score": float(z_score),
                "old_mean": float(old_mean),
                "new_mean": float(new_mean)
            }
            
        return {
            "drift_detected": drift_detected,
            "details": drift_report
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Example usage: check against a dummy new batch
    # Generate dummy new batch for testing
    from sklearn.datasets import make_classification
    X, _ = make_classification(n_samples=100, n_features=30, random_state=99) # Different random state
    columns = [f"V{i+1}" for i in range(28)] + ["Time", "Amount"]
    df_new = pd.DataFrame(X, columns=columns)
    df_new.to_csv("data/raw/new_batch.csv", index=False)
    
    print(detect_drift("data/raw/new_batch.csv"))
