import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from utils.preprocess import load_and_preprocess

def train_model():
# Veriyi yükle ve ön işle
    df = load_and_preprocess("data/creditcard.csv")
    # Modeli oluştur ve eğit
    model = IsolationForest(contamination=0.001, random_state=42)
    model.fit(df.drop("Class", axis=1))

    # Anomali tahminleri
    df["anomaly"] = model.predict(df.drop("Class", axis=1))
    df["anomaly"] = df["anomaly"].apply(lambda x: 1 if x == -1 else 0)

    # Modeli .pkl olarak kaydet
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/anomaly_model.pkl")

    # Değerlendirme metrikleri
    report = classification_report(df["Class"], df["anomaly"])
    matrix = confusion_matrix(df["Class"], df["anomaly"])

    # Metin olarak kaydet
    os.makedirs("results", exist_ok=True)
    with open("results/evaluation.txt", "w") as f:
        f.write(report)

    # Confusion Matrix görselini kaydet
    sns.heatmap(matrix, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    print("[INFO] Eğitim başlıyor...")
    train_model()
    print("[INFO] Eğitim tamamlandı.")