import os
import numpy as np
from src.preprocessing import load_and_preprocess
from src.features import extract_features
from src.classifiers import train_and_evaluate
from src.classifiers import save_model
# Folder where you put the downloaded .edf files
DATA_DIR = os.path.join("data", "raw")

def main():
    print("--- Starting NeuroID System ---")
    
    all_features = []
    all_labels = []
    
    # 1. Find all .edf files
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.edf')]
    if not files:
        print("ERROR: No files found in data/raw/")
        return
        
    print(f"Found {len(files)} subjects. Processing...")

    # 2. Loop through each subject file
    for filename in files:
        file_path = os.path.join(DATA_DIR, filename)
        subject_id = filename[:4] # e.g., S001
        
        try:
            # Step A: Preprocess
            epochs = load_and_preprocess(file_path)
            
            # Step B: Feature Extraction
            feats = extract_features(epochs)
            
            # Step C: Create Labels
            labels = [subject_id] * len(feats)
            
            all_features.append(feats)
            all_labels.extend(labels)
            print(f"  Processed {subject_id}")
            
        except Exception as e:
            print(f"  Skipping {filename}: {e}")

    # 3. Stack everything into big Arrays
    if not all_features:
        print("No features extracted.")
        return

    X = np.vstack(all_features)
    y = np.array(all_labels)
    
    print(f"\nFinal Data Shape: {X.shape}")
    
    # 4. Train AI and Print Result
    print("Training SVM Classifier...")
    
    # CHANGE IS HERE: Unpack two values (accuracy AND model)
    accuracy, model = train_and_evaluate(X, y)
    
    print("="*40)
    print(f"FINAL IDENTIFICATION ACCURACY: {accuracy*100:.2f}%")
    print("="*40)

    # 5. Save the Model
    # Now 'model' exists, so this will work!
    from src.classifiers import save_model
    print("Saving Trained Model...")
    save_model(model, "neuro_id_model.pkl")

if __name__ == "__main__":
    main()