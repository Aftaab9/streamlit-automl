from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    return metrics

def get_risk_scores(model, X_test):
    probs = model.predict_proba(X_test)
    risk_df = pd.DataFrame(probs, columns=[f"Class_{i}" for i in range(probs.shape[1])])
    risk_df["Predicted"] = model.predict(X_test)
    return risk_df
