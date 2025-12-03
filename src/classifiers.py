import joblib
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

def train_and_evaluate(X, y):
    # 1. Create the AI Model
    model = make_pipeline(StandardScaler(), SVC(kernel='linear'))

    # 2. Test using Cross-Validation (Get Accuracy)
    cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv)
    
    # 3. CRITICAL STEP: Train on the FULL dataset for the App
    # (Cross validation splits data, but for the App we want the model to know EVERYONE)
    model.fit(X, y)
    
    # Return BOTH accuracy and the fitted model
    return scores.mean(), model

def save_model(model, filename="svm_model.pkl"):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")