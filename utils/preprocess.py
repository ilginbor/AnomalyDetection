def load_and_preprocess(csv_path):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv(csv_path)

    # Eksik veri kontrolü
    if df.isnull().sum().any():
        df = df.dropna()

    # Amount sütununu ölçekle
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
    df = df.drop(['Time', 'Amount'], axis=1)

    return df