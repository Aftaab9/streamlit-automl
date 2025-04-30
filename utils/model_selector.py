from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np

def select_best_model(X, y):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVM": SVC(probability=True)
    }

    best_score = -np.inf
    best_model = None
    best_name = ""

    for name, model in models.items():
        try:
            score = cross_val_score(model, X, y, cv=3, scoring="accuracy").mean()
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        except Exception as e:
            print(f"Model {name} failed: {e}")

    best_model.fit(X, y)
    return best_model, best_name
