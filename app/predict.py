import joblib
import numpy as np

def predict_transaction(input_features):
    """
    Verilen işlem özelliklerine göre eğitilmiş model ile anomali tespiti yapar.
    
    Parametre:
        input_features: numpy array (tek satır, 29 özellik içermeli)

    Dönüş:
        "Anormal" veya "Normal" şeklinde tahmin sonucu (string)
    """
    # Eğitilmiş Isolation Forest modelini yükle (model/anomaly_model.pkl dosyasından)
    model = joblib.load("model/anomaly_model.pkl")

    # Model ile tahmin yap (-1: anormal, 1: normal)
    prediction = model.predict(input_features)

    # Tahmin sonucunu insan okunabilir hale getir
    return "Anormal" if prediction[0] == -1 else "Normal"


# Bu dosya doğrudan çalıştırılırsa kullanıcıdan CLI üzerinden veri alınıp tahmin yapılır
if __name__ == "__main__":
    print("📥 Kullanıcıdan işlem bilgisi alınıyor...")

    # Kullanıcının 29 özellik girmesi beklenir (V1–V28 + Amount gibi)
    # Bu veri normalde GUI veya başka bir sistemden de alınabilir
    try:
        # Kullanıcıdan virgül ile ayrılmış 29 sayı al
        values = input("Lütfen 29 sayı girin (virgül ile ayrılmış):\n").split(",")

        # Sayısal değerlere dönüştürüp NumPy dizisine çevir, ardından şekillendir
        values = np.array([float(x.strip()) for x in values]).reshape(1, -1)

        # Tahmin fonksiyonunu çağır
        result = predict_transaction(values)

        # Sonucu yazdır
        print(f"\n🔎 Sonuç: {result}")

    # Hatalı girişler durumunda kullanıcıya bilgi ver
    except Exception as e:
        print(f"❌ Hata: {e}")
