from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


def model_training(X, y, model_name, model_path, cv=5):
    #model initialization
    models={
        "logistic regression": LogisticRegression(),
        "random forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0),
        "SVM": SVC(kernel="rbf", C=1, gamma="scale", probability=True)
}

    #standardize features
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)

    #get the model
    model=models[model_name]

    auc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")

    #train on dataset
    model.fit(X_scaled, y)
    #save model & scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, model_path.replace(".pkl", "_scaler.pkl"))

    print(f"model {model_name} saved to {model_path}")
    print(f"scaler saved to {model_path.replace(".pkl", "_scaler.pkl")}")
    print(f"✅ Cross-validation AUC-ROC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
    return model, scaler, auc_scores