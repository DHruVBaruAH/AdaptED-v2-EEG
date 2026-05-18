import pandas as pd

df = pd.read_csv("data/features/eeg_features.csv")

# Channel index pairs (left, right) based on CHANNELS list order
pairs = [
    (2, 3),   # F3, F4
    (4, 5),   # C3, C4
    (6, 7),   # P3, P4
    (8, 9),   # O1, O2
    (10, 11), # F7, F8
    (12, 13), # T7, T8
    (14, 15), # P7, P8
]

band_names = ["delta", "theta", "alpha", "beta", "gamma"]

for left_idx, right_idx in pairs:
    for band_idx, band_name in enumerate(band_names):
        left_col = f"pow_freq_bands_ch{left_idx}_band{band_idx}"
        right_col = f"pow_freq_bands_ch{right_idx}_band{band_idx}"
        if left_col in df.columns and right_col in df.columns:
            df[f"asym_ch{left_idx}_ch{right_idx}_{band_name}"] = (
                df[left_col] - df[right_col]
            )

print(f"Done. New shape: {df.shape}")
df.to_csv("data/features/eeg_features.csv", index=False)