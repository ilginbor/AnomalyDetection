def load_and_preprocess(csv_path):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # 📌 1. Veri setini CSV dosyasından oku
    df = pd.read_csv(csv_path)

    # 📌 2. Eksik verileri kontrol et ve varsa sil
    if df.isnull().sum().any():
        df = df.dropna()  # Not: Projede eksik veri olmasa da güvenlik önlemi olarak var

    # 📌 3. 'Amount' sütununu standart ölçekleme ile normalize et (ortalama 0, std 1)
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df[['Amount']])  # Yeni sütun olarak eklenir

    # 📌 4. 'Time' ve orijinal 'Amount' sütunlarını kaldır
    df = df.drop(['Time', 'Amount'], axis=1)

    # 📌 5. Temizlenmiş ve ölçeklenmiş veri çerçevesini döndür
    return df
