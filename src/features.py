import numpy as np
from scipy.signal import welch

def extract_features(epochs):
    # Get the raw numbers: (Windows, Channels, TimePoints)
    data = epochs.get_data(copy=True)
    n_epochs, n_channels, n_times = data.shape
    
    features_list = []
    fs = 160  # Sampling frequency is 160Hz for this dataset

    for i in range(n_epochs):
        single_window = data[i] # Shape: (12, 1600)
        window_features = []

        # --- FEATURE 1: FWP (Baseline from Paper) ---
        # Convert Time -> Frequency (Power Spectral Density)
        freqs, psd = welch(single_window, fs=fs, nperseg=fs*2)
        
        # Calculate FWP for Beta (13-30Hz) and Gamma (30-50Hz)
        for low, high in [(13, 30), (30, 50)]:
            # Find which frequencies are in this range
            idx = np.logical_and(freqs >= low, freqs <= high)
            
            # Formula: Sum(Power * Frequency)
            fwp = np.sum(psd[:, idx] * freqs[idx], axis=1)
            window_features.extend(fwp)

        # --- FEATURE 2: Connectivity (Your Novelty) ---
        # Correlation Matrix (12x12)
        corr_matrix = np.corrcoef(single_window)
        
        # Take only the top triangle (unique connections)
        unique_connections = corr_matrix[np.triu_indices(n_channels, k=1)]
        window_features.extend(unique_connections)

        features_list.append(window_features)

    return np.array(features_list)