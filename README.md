# 🛡️ Anomaly Detection in Financial Transactions

Bu proje, kredi kartı işlemleri üzerinde anomali tespiti yapmayı amaçlar. Makine öğrenmesi teknikleriyle finansal dolandırıcılıkların tespiti hedeflenmiştir.

## 🚀 Proje Özeti

- Veri seti: `data/creditcard.csv` (Kaynak: Kaggle)
- Algoritma: Isolation Forest (scikit-learn)
- Arayüz: CLI & CustomTkinter GUI
- Çıktılar: Confusion Matrix, classification report, .pkl model dosyası

## 📦 Gerekli Kütüphaneler
Projeyi çalıştırmadan önce gerekli Python kütüphanelerini kurmak için
terminalde şu komutu çalıştırabilirsiniz :

`pip install pandas numpy scikit-learn matplotlib seaborn customtkinter`

## ⚙️ Kullanım

VS Code terminalinden:

1. Model eğitimi:
python -m model.train_model

2. CLI ile tahmin:
python app/predict.py

3. GUI başlatmak için:
python app/gui_app.py

4. Hepsini tek yerden yönetmek için:
python main.py

Notlar:
data/creditcard.csv dosyası büyük boyut nedeniyle GitHub’a yüklenmemiştir .gitignore içinde tutulmaktadır.

Orijinal veri setine https://www.kaggle.com/datasets/whenamancodes/fraud-detection linkinden ulaşabilirsiniz.
