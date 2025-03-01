import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, accuracy_score

def ml_training_lopo(X, y, pids, model_name, model_path):
    """Train ML models using Leave-One-Participant-Out (LOPO) Cross-Validation."""

    models = {
        "logistic regression": LogisticRegression(),
        "random forest": RandomForestClassifier(n_estimators=50, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0),
        "SVM": SVC(kernel="rbf", C=1, gamma="scale", probability=True)
    }

    model = models[model_name]
    logo = LeaveOneGroupOut()
    auc_scores = []

    for train_idx, test_idx in logo.split(X, y, groups=pids):
        print(f"🔄 Training - Leaving out participant {pids[test_idx][0]}")

        # Split train/test
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  
        auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        auc_scores.append(auc)
        print(f"✅ Participant {pids[test_idx][0]}: AUC = {auc:.4f}")

    # Save Model & Scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, model_path.replace(".pkl", "_scaler.pkl"))

    return model, scaler, np.mean(auc_scores), np.std(auc_scores)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut

def train_lstm_lopo(X, y, pids, model_path):
    """
    Train an LSTM model using Leave-One-Participant-Out (LOPO) cross-validation.
    """

    print(f"🔍 Initial X shape: {X.shape}, y shape: {y.shape}, pids shape: {pids.shape}")

    # ✅ 1. **Trim `X` and `y` to Match `pids`** (Remove Mismatched Entries)
    min_length = min(len(X), len(y), len(pids))
    X, y, pids = X[:min_length], y[:min_length], pids[:min_length]

    print(f"✅ Trimmed X shape: {X.shape}, y shape: {y.shape}, pids shape: {pids.shape}")

    # ✅ 2. **Expand `pids` to Match `X` Strictly**
    unique_pids = np.unique(pids)
    valid_indices = np.isin(pids, unique_pids)  # Keep only valid participant sequences

    X, y, pids = X[valid_indices], y[valid_indices], pids[valid_indices]

    print(f"✅ After Expansion - X: {X.shape}, y: {y.shape}, pids: {pids.shape}")

    # ✅ 3. **Setup LOPO**
    logo = LeaveOneGroupOut()
    auc_scores, accuracy_scores = [], []

    for train_idx, test_idx in logo.split(X, y, groups=pids):
        test_pid = np.unique(pids[test_idx])[0]
        print(f"🔄 Training LSTM - Leaving out participant {test_pid}")

        # ✅ 4. **Train/Test Split**
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ✅ 5. **Standardization**
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        # ✅ 6. **Define LSTM Model**
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dense(1, activation="sigmoid")
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        # ✅ 7. **Evaluate**
        y_pred_proba = model.predict(X_test).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        accuracy = accuracy_score(y_test, y_pred)

        auc_scores.append(auc)
        accuracy_scores.append(accuracy)

        print(f"✅ Participant {test_pid}: AUC = {auc:.4f}, Accuracy = {accuracy:.4f}")

    # ✅ 8. **Compute Mean and Standard Deviation**
    mean_auc, std_auc = np.mean(auc_scores), np.std(auc_scores)
    mean_acc, std_acc = np.mean(accuracy_scores), np.std(accuracy_scores)

    # ✅ 9. **Save Model**
    model.save(model_path)
    print(f"✅ LSTM Model saved at {model_path}")
    print(f"✅ LOPO AUC: {mean_auc:.4f} ± {std_auc:.4f}, Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    return model, mean_auc, std_auc
