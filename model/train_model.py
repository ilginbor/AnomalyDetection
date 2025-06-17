import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from utils.preprocess import load_and_preprocess  # Veriyi yükleyip ölçeklendiren ve işleyen fonksiyon

def train_model():
    # 📌 1. Veri Yükleme ve Ön İşleme
    df = load_and_preprocess("data/creditcard.csv")  # Veriyi oku ve hazırla (ölçekleme + class dengesi)

    # 📌 2. Isolation Forest modeli oluşturuluyor
    model = IsolationForest(contamination=0.001, random_state=42)  # %0.1 oranında anomali varsayımı
    model.fit(df.drop("Class", axis=1))  # Sınıf etiketi hariç tüm verilerle eğit

    # 📌 3. Tahmin yapılır ve -1 → 1 (anormal), 1 → 0 (normal) olarak etiketlenir
    df["anomaly"] = model.predict(df.drop("Class", axis=1))        # Tahmin sonucu: -1 veya 1
    df["anomaly"] = df["anomaly"].apply(lambda x: 1 if x == -1 else 0)  # -1 → 1 (anormal), 1 → 0 (normal)

    # 📌 4. Eğitilen modeli 'model/' klasörüne kaydet
    os.makedirs("model", exist_ok=True)  # model klasörü yoksa oluştur
    joblib.dump(model, "model/anomaly_model.pkl")  # model kaydı

    # 📌 5. Model Performansı: precision, recall, f1-score, accuracy vs.
    report = classification_report(df["Class"], df["anomaly"])  # Gerçek vs tahmin etiketleri
    matrix = confusion_matrix(df["Class"], df["anomaly"])       # Karışıklık matrisi

    # 📌 6. Değerlendirme çıktısı metin olarak kaydedilir
    os.makedirs("results", exist_ok=True)
    with open("results/evaluation.txt", "w") as f:
        f.write(report)

    # 📌 7. Confusion Matrix görselleştirilip PNG olarak kaydedilir
    sns.heatmap(matrix, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    plt.close()

# Bu dosya doğrudan çalıştırıldığında eğitim işlemi başlatılır
if __name__ == "__main__":
    print("[INFO] Eğitim başlıyor...")
    train_model()
    print("[INFO] Eğitim tamamlandı.")
