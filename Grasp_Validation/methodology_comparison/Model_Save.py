import pickle
from datetime import datetime

def save_model_with_timestamp(model, base_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if hasattr(model, 'save'):
        # Keras model
        filename = f"{base_name}_{timestamp}.keras"
        model.save(filename)
    else:
        # scikit-learn model (LogisticRegression, DecisionTreeClassifier, MLPClassifier, ...)
        filename = f"{base_name}_{timestamp}.joblib"
        from joblib import dump
        dump(model, filename)
    print(f"Model saved to {filename}")