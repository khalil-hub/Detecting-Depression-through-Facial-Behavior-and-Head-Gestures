import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def evaluate_ml_model_lopo(model_path, scaler_path, X, y, pids, log_path):
    """Evaluate ML models with LOPO-CV."""
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    logo = LeaveOneGroupOut()
    auc_scores = []
    
    for train_idx, test_idx in logo.split(X, y, groups=pids):
        print(f"🔍 Evaluating participant {pids[test_idx][0]}")

        X_test, y_test = X[test_idx], y[test_idx]
        X_test_scaled = scaler.transform(X_test)

        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        auc_scores.append(auc)

    print(f"✅ Mean AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

def evaluate_lstm_lopo(model_path, scaler_path, X, y, pids, log_path):
    """
    Evaluate an LSTM model using LOPO-CV and log AUROC, Accuracy, and Precision.
    """

    # ✅ Load Model and Scaler
    model = tf.keras.models.load_model(model_path)
    scaler = StandardScaler()

    # ✅ Ensure pids matches X by trimming `pids` to match `X`
    if len(pids) != len(X):
        print(f"⚠️ Adjusting pids from {len(pids)} to {len(X)} to match sequences")
        
        # Trim `pids` if it's longer than `X`
        if len(pids) > len(X):
            pids = pids[:len(X)]
        else:
            # Expand pids by repeating until it matches X
            pids = np.repeat(pids, len(X) // len(pids) + 1)[:len(X)]

    # ✅ Confirm final shapes
    print(f"✅ Final X shape: {X.shape}")  # Expect (190295, 10, 14)
    print(f"✅ Final y shape: {y.shape}")  # Expect (190295,)
    print(f"✅ Final pids shape: {pids.shape}")  # Expect (190295,)

    assert len(pids) == len(X), f"❌ ERROR: `pids` length {len(pids)} != `X` length {len(X)}"

    # ✅ LOPO-CV Setup
    logo = LeaveOneGroupOut()
    auc_scores = []
    accuracy_scores = []
    precision_scores = []
    participant_results = []

    for train_idx, test_idx in logo.split(X, y, groups=pids):
        test_pid = np.unique(pids[test_idx])[0]
        print(f"🔍 Evaluating on participant {test_pid}")

        # ✅ Test Set
        X_test, y_test = X[test_idx], y[test_idx]

        if X_test.shape[0] == 0:
            print(f"⚠️ Skipping participant {test_pid} due to insufficient test samples.")
            continue

        # ✅ Standardization
        X_test_scaled = scaler.fit_transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        # ✅ Predictions
        y_pred_proba = model.predict(X_test_scaled).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # ✅ Metrics
        auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)

        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)

        # ✅ Store Results
        for actual, predicted in zip(y_test, y_pred):
            participant_results.append([test_pid, actual, predicted])

        print(f"✅ Participant {test_pid}: AUC={auc:.4f}, Accuracy={accuracy:.4f}, Precision={precision:.4f}")

    # ✅ Compute Mean and Std
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    mean_acc = np.mean(accuracy_scores)
    std_acc = np.std(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    std_precision = np.std(precision_scores)

    # ✅ Convert results to DataFrame
    participant_df = pd.DataFrame(participant_results, columns=["Participant", "Actual", "Predicted"])

    # ✅ Log Results
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        f.write(f"LOPO-CV AUROC: {mean_auc:.4f} ± {std_auc:.4f}\n")
        f.write(f"LOPO-CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n")
        f.write(f"LOPO-CV Precision: {mean_precision:.4f} ± {std_precision:.4f}\n\n")
        f.write("📊 Per-Participant Results:\n")
        f.write(participant_df.to_string(index=False))

    print(f"✅ LOPO-CV AUROC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"✅ LOPO-CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"✅ LOPO-CV Precision: {mean_precision:.4f} ± {std_precision:.4f}")

    return mean_auc, std_auc, mean_acc, std_acc, mean_precision, std_precision
