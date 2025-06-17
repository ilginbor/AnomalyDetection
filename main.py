def main():
    import sys

    # Başlık ve menü seçenekleri
    print("🛡️ Anomaly Detection in Financial Transactions")
    print("1 - Modeli Eğit")               # Eğitim işlemi (Isolation Forest)
    print("2 - CLI ile Tahmin Yap")        # Komut satırı üzerinden tahmin
    print("3 - GUI Arayüzü Başlat")        # Grafik arayüz başlatılır
    secim = input("Seçiminiz: ")           # Kullanıcıdan seçim alınır

    # Seçenek 1: Modeli eğit
    if secim == "1":
        from model.train_model import train_model
        train_model()                      # Isolation Forest algoritmasıyla modeli eğit
        print("✅ Eğitim tamamlandı.")     # Eğitim tamamlandı mesajı

    # Seçenek 2: CLI (komut satırı) ile anlık tahmin yap
    elif secim == "2":
        from app.predict import predict_transaction
        import numpy as np
        giris = input("29 özellik girin (virgülle): ").split(",")  # 29 özellik alınır
        veriler = np.array([float(x.strip()) for x in giris]).reshape(1, -1)  # NumPy ile şekillendirilir
        sonuc = predict_transaction(veriler)                      # Tahmin fonksiyonu çağrılır
        print(f"🔎 Tahmin sonucu: {sonuc}")                       # Tahmin sonucu yazdırılır

    # Seçenek 3: GUI (grafik arayüz) başlat
    elif secim == "3":
        import subprocess
        subprocess.run(["python", "-m", "app.gui_app"])           # GUI uygulaması başlatılır

    # Geçersiz seçim durumu
    else:
        print("❌ Geçersiz seçim.")

# main fonksiyonu doğrudan çalıştırıldığında uygulamayı başlat
if __name__ == "__main__":
    main()
