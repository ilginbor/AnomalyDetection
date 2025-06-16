def main():
    import sys
    print("🛡️ Anomaly Detection in Financial Transactions")
    print("1 - Modeli Eğit")
    print("2 - CLI ile Tahmin Yap")
    print("3 - GUI Arayüzü Başlat")
    secim = input("Seçiminiz: ")

    if secim == "1":
        from model.train_model import train_model
        train_model()
        print("✅ Eğitim tamamlandı.")

    elif secim == "2":
        from app.predict import predict_transaction
        import numpy as np
        giris = input("29 özellik girin (virgülle): ").split(",")
        veriler = np.array([float(x.strip()) for x in giris]).reshape(1, -1)
        sonuc = predict_transaction(veriler)
        print(f"🔎 Tahmin sonucu: {sonuc}")

    elif secim == "3":
        import subprocess
        subprocess.run(["python", "-m", "app.gui_app"])
    else:
        print("❌ Geçersiz seçim.")

if __name__ == "__main__":
    main()