import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_excel("C:/Users/enesa/OneDrive/Desktop/Ders/merc.xlsx")



df = pd.get_dummies(df, columns=['transmission'], drop_first=True)

#  3. Bağımsız ve Bağımlı Değişkenleri Belirle
X = df.drop(columns=['price'])  # Girdi değişkenleri
y = df['price']  # Hedef değişken

# Hedef değişkeni sınıflara ayıralım (örneğin, fiyatı 20,000'den yüksek ve düşük olarak sınıflandıralım)
y = (y > 20000).astype(int)

#  4. Veriyi Eğitim ve Test Setlerine Ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  5. Random Forest Modelini Ölçeklendirmeden Eğit
rf_model_no_scaling = RandomForestClassifier(random_state=42)
rf_model_no_scaling.fit(X_train, y_train)

#  6. Ölçeklendirmeden Tahmin Yap ve Değerlendir
y_pred_no_scaling = rf_model_no_scaling.predict(X_test)

accuracy_no_scaling = accuracy_score(y_test, y_pred_no_scaling)
conf_matrix_no_scaling = confusion_matrix(y_test, y_pred_no_scaling)

#  7. Veriyi Ölçeklendir
scaler_X = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

#  8. Random Forest Modelini Ölçeklendirilmiş Verilerle Eğit
rf_model_scaled = RandomForestClassifier(random_state=42)
rf_model_scaled.fit(X_train_scaled, y_train)

# 📌 9. Ölçeklendirilmiş Tahmin Yap ve Değerlendir
y_pred_scaled = rf_model_scaled.predict(X_test_scaled)

accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
conf_matrix_scaled = confusion_matrix(y_test, y_pred_scaled)

#  10. Sonuçları Yazdır
print(" Ölçeklendirilmemiş Verilerle Sonuçlar:")
print(f"Accuracy: {accuracy_no_scaling:.4f}")
print(f"Confusion Matrix:\n{conf_matrix_no_scaling}\n")

print(" Ölçeklendirilmiş Verilerle Sonuçlar:")
print(f"Accuracy: {accuracy_scaled:.4f}")
print(f"Confusion Matrix:\n{conf_matrix_scaled}\n")

#  11. Confusion Matrix'i Grafik Olarak Çizdir
def plot_confusion_matrix(conf_matrix, title='Confusion Matrix'):
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Low Price', 'High Price'], yticklabels=['Low Price', 'High Price'])
    plt.title(title)
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.show()

#  12. Grafik Olarak Görselleştir
plot_confusion_matrix(conf_matrix_no_scaling, title="Confusion Matrix (Random Forest - Ölçeklendirilmemiş Veri)")
plot_confusion_matrix(conf_matrix_scaled, title="Confusion Matrix (Random Forest - Ölçeklendirilmiş Veri)")
