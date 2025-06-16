import customtkinter as ctk
import numpy as np

from app.predict import predict_transaction

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class AnomalyApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Anomaly Detection")
        self.geometry("400x400")
        self.amount_entry = ctk.CTkEntry(self, placeholder_text="Amount")
        self.amount_entry.pack(pady=10)

        self.v1_entry = ctk.CTkEntry(self, placeholder_text="V1")
        self.v1_entry.pack(pady=10)

        self.v2_entry = ctk.CTkEntry(self, placeholder_text="V2")
        self.v2_entry.pack(pady=10)

        self.v3_entry = ctk.CTkEntry(self, placeholder_text="V3")
        self.v3_entry.pack(pady=10)

        self.result_label = ctk.CTkLabel(self, text="", text_color="white")
        self.result_label.pack(pady=10)

        self.predict_button = ctk.CTkButton(self, text="Tahmin Et", command=self.predict)
        self.predict_button.pack(pady=10)

    def predict(self):
        try:
            amt = float(self.amount_entry.get())
            v1 = float(self.v1_entry.get())
            v2 = float(self.v2_entry.get())
            v3 = float(self.v3_entry.get())

            # 29 feature: [scaled_amount, V1, ..., V28]
            features = np.zeros((1, 29))
            features[0, 0] = amt     # scaled_amount
            features[0, 1] = v1      # V1
            features[0, 2] = v2      # V2
            features[0, 3] = v3      # V3
            # Diğerleri 0

            result = predict_transaction(features)
            self.result_label.configure(text=f"Sonuç: {result}")

        except Exception as e:
            self.result_label.configure(text=f"Hata: {e}")


if __name__ == "__main__":
    app = AnomalyApp()
    app.mainloop()