import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_excel("C:/Users/enesa/OneDrive/Desktop/Ders/merc.xlsx")


df = pd.get_dummies(df, columns=['transmission'], drop_first=True)


X = df.drop(columns=['price'])  # Girdi değişkenleri
y = df['price']  # Hedef değişken

y = (y > 20000).astype(int)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


knn_model_no_scaling = KNeighborsClassifier(n_neighbors=5)
knn_model_no_scaling.fit(X_train, y_train)

y_pred_no_scaling = knn_model_no_scaling.predict(X_test)

accuracy_no_scaling = accuracy_score(y_test, y_pred_no_scaling)
conf_matrix_no_scaling = confusion_matrix(y_test, y_pred_no_scaling)

scaler_X = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

knn_model_scaled = KNeighborsClassifier(n_neighbors=5)
knn_model_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = knn_model_scaled.predict(X_test_scaled)

accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
conf_matrix_scaled = confusion_matrix(y_test, y_pred_scaled)

print("🔴 Ölçeklendirilmemiş Verilerle Sonuçlar:")
print(f"Accuracy: {accuracy_no_scaling:.4f}")
print(f"Confusion Matrix:\n{conf_matrix_no_scaling}\n")

print("🟢 Ölçeklendirilmiş Verilerle Sonuçlar:")
print(f"Accuracy: {accuracy_scaled:.4f}")
print(f"Confusion Matrix:\n{conf_matrix_scaled}\n")


def plot_confusion_matrix(conf_matrix, title='Confusion Matrix'):
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Low Price', 'High Price'], yticklabels=['Low Price', 'High Price'])
    plt.title(title)
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.show()

plot_confusion_matrix(conf_matrix_no_scaling, title="Confusion Matrix (KNN - Ölçeklendirilmemiş Veri)")
plot_confusion_matrix(conf_matrix_scaled, title="Confusion Matrix (KNN - Ölçeklendirilmiş Veri)")
