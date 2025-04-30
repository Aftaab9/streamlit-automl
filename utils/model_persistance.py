import joblib
import os

def save_model(model, model_name="best_model.pkl", directory="saved_models"):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, model_name)
    joblib.dump(model, filepath)
    return filepath

def load_model(filepath):
    return joblib.load(filepath)
