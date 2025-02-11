from sklearn.model_selection import cross_val_score
import os
import joblib

def evaluate_model(model_path, scaler_path, X, y, log_path, cv=5):

    # Load model & scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Apply the same transformation to X
    X_scaled = scaler.transform(X)

    # Perform cross-validation
    auc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")

    # Log results
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        f.write(f"Cross-Validation AUC-ROC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}\n")

    print(f"✅ Cross-validation AUC-ROC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
    print(f"✅ Evaluation metrics logged to {log_path}")

    return auc_scores.mean(), auc_scores.std()

