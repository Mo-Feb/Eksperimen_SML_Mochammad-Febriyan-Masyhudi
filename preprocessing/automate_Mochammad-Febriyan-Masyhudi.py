import json
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(BASE_DIR, "bank_model_metadata.json")
INPUT_PATH = os.path.join(BASE_DIR, "..", "bank_leads_raw", "bank-raw.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "bank_leads_preprocessing")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "bank_preprocessing.csv")

def load_metadata():
    with open(METADATA_PATH, "r") as f:
        return json.load(f)

def clean_and_feature_engineer(df, metadata):
    df = df.copy()
    target = None
    if 'deposit' in df.columns:
        df['deposit'] = df['deposit'].map({'yes': 1, 'no': 0})
        target = df['deposit']

    # 1. QUARTER MAPPING
    mapping_quarter = metadata['mappings']['quarter']
    if 'month' in df.columns:
        df['quarter'] = df['month'].map(mapping_quarter).fillna('unknown')

    # 2. PREV CONTACT RESULT
    mapping_poutcome = metadata['mappings']['poutcome']
    if 'poutcome' in df.columns:
        df['prev_contact_result'] = df['poutcome'].map(mapping_poutcome).fillna(0).astype(int)

    # 3. IS PREV CONTACTED
    if 'pdays' in df.columns:
        df['is_prev_contacted'] = np.where(df['pdays'] != -1, 'yes', 'no')

    # 4. DEBT STATUS
    housing_val = df['housing'].map({'yes': 1, 'no': 0}) if 'housing' in df.columns else 0
    loan_val = df['loan'].map({'yes': 1, 'no': 0}) if 'loan' in df.columns else 0
    df['debt_status'] = housing_val + loan_val

    # 5. AGE GROUP (Binning)
    if 'age' in df.columns:
        bins = metadata['binning']['age_bins']
        labels = metadata['binning']['age_labels']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
        df['age_group'] = df['age_group'].astype(str)

    # 6. CAPPING OUTLIERS UNTUK CAMPAIGN & PREVIOUS
    capping = metadata.get('capping', {})
    for col, value in capping.items():
        if col in df.columns:
            df[col] = np.where(df[col] > value, value, df[col])

    # 7. SCALING UNTUK BALANCE, CAMPAIGN, PREVIOUS
    scaler_info = metadata.get('scaler', {})
    if scaler_info:
        scaler_mean = scaler_info['mean']
        scaler_scale = scaler_info['scale']
        scaling_features = ['balance', 'campaign', 'previous']
        for idx, feature in enumerate(scaling_features):
            if feature in df.columns:
                df[feature] = (df[feature] - scaler_mean[idx]) / scaler_scale[idx]

    # 8. DROP UNUSED COLUMNS
    cols_to_drop = metadata.get('features_to_drop', [])
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # 9. ONE-HOT ENCODING
    df_encoded = pd.get_dummies(df)
    expected_columns = metadata.get('model_input_columns', [])
    if expected_columns:
        for col in expected_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        df_encoded = df_encoded[expected_columns]

    if target is not None:
        df_encoded['deposit'] = target.values

    return df_encoded

def run_automation():
    metadata = load_metadata()
    if metadata is None: return
    
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Dataset mentah tidak ditemukan di {INPUT_PATH}")
        return

    df_raw = pd.read_csv(INPUT_PATH, sep=None, engine='python')
    
    df_clean = clean_and_feature_engineer(df_raw, metadata)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_clean.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Berhasil! Data bersih disimpan di: {OUTPUT_FILE}")
    print(f"Ukuran Data Akhir: {df_clean.shape}")

if __name__ == "__main__":
    run_automation()