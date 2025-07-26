from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess_data(df):
    # Drop ID column if present
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    
    # Fill missing values if any (simple fill with median)
    df.fillna(df.median(), inplace=True)
    
    # Separate features and target
    X = df.drop(columns=['default.payment.next.month'])
    y = df['default.payment.next.month']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler
