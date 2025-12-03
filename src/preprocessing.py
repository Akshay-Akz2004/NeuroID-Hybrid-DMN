import mne

# The 12 channels used in the paper for Default Mode Network
DMN_CHANNELS = ['C1', 'TP8', 'FC5', 'F8', 'FT8', 'AF3', 'T8', 'P3', 'Iz', 'O2', 'T9', 'TP7']

def load_and_preprocess(file_path):
    # 1. Load Data (verbose=False hides the messy logs)
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    
    # 2. Fix channel names (PhysioNet sometimes has dots like 'FC5.')
    raw.rename_channels(lambda x: x.strip('.'))
    
    # 3. Select only the 12 channels we need
    # We use a list comprehension to check which ones exist in the file
    picks = [ch for ch in DMN_CHANNELS if ch in raw.ch_names]
    raw.pick_channels(picks)
    
    # 4. Filter the signal (0.5Hz to 50Hz) to remove noise
    raw.filter(0.5, 50.0, verbose=False)
    
    # 5. Cut into 10-second non-overlapping windows
    # duration=10.0 matches the paper's protocol
    # overlap=5.0 means the windows slide: 0-10s, 5-15s, 10-20s...
# This DOUBLES your training data without downloading anything new.
    epochs = mne.make_fixed_length_epochs(raw, duration=10.0, overlap=5.0, verbose=False)
    
    return epochs