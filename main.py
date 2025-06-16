# 🛡️ Anomaly Detection in Financial Transactions (VS Code ile)

# 1. Gerekli kütüphaneler
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Veri setini oku
df = pd.read_csv('creditcard.csv')
print("İlk 5 satır:")
print(df.head())

# 3. Veri ön işleme
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
df = df.drop(['Time', 'Amount'], axis=1)

# 4. Isolation Forest modeli
model = IsolationForest(contamination=0.001, random_state=42)
model.fit(df.drop('Class', axis=1))

df['anomaly'] = model.predict(df.drop('Class', axis=1))
df['anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)

# 5. Değerlendirme
print("\nSınıflandırma Raporu:")
print(classification_report(df['Class'], df['anomaly']))
print("Confusion Matrix:")
print(confusion_matrix(df['Class'], df['anomaly']))

# 6. Grafikle göster
sns.heatmap(confusion_matrix(df['Class'], df['anomaly']), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.show()
