import joblib
import numpy as np

def predict_transaction(input_features):
    """
    input_features: numpy array (tek satır, işlem özellikleri)
    Örnek: np.array([[...]])
    """
    # Eğitilmiş modeli yükle
    model = joblib.load("model/anomaly_model.pkl")
    # Tahmin yap (-1 = anormal, 1 = normal → 1 → anormal)
    prediction = model.predict(input_features)
    return "Anormal" if prediction[0] == -1 else "Normal"


if __name__ == "__main__":
    print("📥 Kullanıcıdan işlem bilgisi alınıyor...")
    # Örnek olarak scaled_amount dahil, V1 - V28 olmak üzere 29 özellik girilmelidir
    # Gerçek projede bu veriler frontend'den veya GUI'den alınabilir

    try:
        values = input("Lütfen 29 sayı girin (virgül ile ayrılmış):\n").split(",")
        values = np.array([float(x.strip()) for x in values]).reshape(1, -1)

        result = predict_transaction(values)
        print(f"\n🔎 Sonuç: {result}")

    except Exception as e:
        print(f"❌ Hata: {e}")