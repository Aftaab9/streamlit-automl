import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        df = df.sort_values("Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Importance", y="Feature", palette="viridis")
        plt.title("Feature Importance")
        plt.tight_layout()
        return plt.gcf()
    return None

def plot_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    return plt.gcf()
