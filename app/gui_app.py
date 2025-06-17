import customtkinter as ctk  # Gelişmiş, modern görünümlü Tkinter arayüzü için CustomTkinter
import numpy as np

from app.predict import predict_transaction  # Eğitilmiş modeli kullanarak tahmin yapma fonksiyonu

# GUI temasını ayarla (karanlık mod, mavi tema)
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Ana GUI uygulama sınıfı tanımlanıyor
class AnomalyApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Anomaly Detection")     # Pencere başlığı
        self.geometry("400x400")            # Pencere boyutu

        # Kullanıcıdan Amount (tutar) girişi al
        self.amount_entry = ctk.CTkEntry(self, placeholder_text="Amount")
        self.amount_entry.pack(pady=10)

        # V1 - V3 arası girişler (sınırlı demo)
        self.v1_entry = ctk.CTkEntry(self, placeholder_text="V1")
        self.v1_entry.pack(pady=10)

        self.v2_entry = ctk.CTkEntry(self, placeholder_text="V2")
        self.v2_entry.pack(pady=10)

        self.v3_entry = ctk.CTkEntry(self, placeholder_text="V3")
        self.v3_entry.pack(pady=10)

        # Tahmin sonucunu gösterecek etiket
        self.result_label = ctk.CTkLabel(self, text="", text_color="white")
        self.result_label.pack(pady=10)

        # "Tahmin Et" butonu ve butona tıklanınca çalışacak fonksiyon bağlantısı
        self.predict_button = ctk.CTkButton(self, text="Tahmin Et", command=self.predict)
        self.predict_button.pack(pady=10)

    # Tahmin fonksiyonu: Kullanıcının girdiği verileri alır ve tahmin yapar
    def predict(self):
        try:
            # Girişleri float’a çevir
            amt = float(self.amount_entry.get())
            v1 = float(self.v1_entry.get())
            v2 = float(self.v2_entry.get())
            v3 = float(self.v3_entry.get())

            # 29 özellikli veri yapısı oluşturuluyor
            # Sadece ilk 4 özellik (Amount, V1, V2, V3) kullanıcıdan alınıyor
            features = np.zeros((1, 29))    # 1 işlem için 29 özelliklik sıfır matrisi
            features[0, 0] = amt            # scaled_amount
            features[0, 1] = v1             # V1
            features[0, 2] = v2             # V2
            features[0, 3] = v3             # V3
            # Kalan özellikler 0 kalır (demo/test için yeterli)

            # Tahmini yap
            result = predict_transaction(features)

            # Sonucu GUI'de göster
            self.result_label.configure(text=f"Sonuç: {result}")

        except Exception as e:
            # Hata durumunda kullanıcıya bilgi ver
            self.result_label.configure(text=f"Hata: {e}")


# Bu dosya doğrudan çalıştırıldığında uygulama başlatılır
if __name__ == "__main__":
    app = AnomalyApp()
    app.mainloop()
