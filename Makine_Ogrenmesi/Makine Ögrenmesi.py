import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest, boxcox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Veriyi yükleme
df = pd.read_excel("C:/Users/enesa/OneDrive/Desktop/Ders/merc.xlsx")
print(df.head())


# Eksik veri kontrolü
print(df.isnull().sum())

# Kategorik değişken: 'transmission'
encoder = OneHotEncoder(sparse_output=False)
transmission_encoded = encoder.fit_transform(df[['transmission']])
transmission_df = pd.DataFrame(transmission_encoded, columns=encoder.get_feature_names_out(['transmission']))
df = pd.concat([df.drop('transmission', axis=1), transmission_df], axis=1)

# Korelasyon matrisi
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Korelasyon Matrisi')
plt.show()

# Normallik testi (price sütunu için)
stat, p = normaltest(df['price'])
print(f'D\'Agostino K^2 Testi: p-değeri = {p}')
if p < 0.05:
    print("Veri normal dağılmamış. Dönüşüm uygulanacak.")
else:
    print("Veri normal dağılıyor.")

# Dönüşümler: Box-Cox, Log, Square Root
df['price_boxcox'], _ = boxcox(df['price'])
df['price_log'] = np.log(df['price'])
df['price_sqrt'] = np.sqrt(df['price'])

# Dönüşüm sonrası normallik testi
transformations = ['price_boxcox', 'price_log', 'price_sqrt']
for col in transformations:
    stat, p = normaltest(df[col])
    print(f'{col} için p-değeri: {p}')

# Bağımlı ve bağımsız değişkenler
X = df.drop(['price', 'price_boxcox', 'price_log', 'price_sqrt'], axis=1)
y = df['price']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model eğitimi (Lineer Regresyon)
model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin ve performans metrikleri
y_pred = model.predict(X_test)
print(f'R² Skoru: {r2_score(y_test, y_pred)}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}')

# Fiyat-Yıl İlişkisi
plt.figure(figsize=(10, 6))
sns.scatterplot(x='year', y='price', data=df)
plt.title('Fiyat-Yıl İlişkisi')
plt.show()

# Dönüşüm sonuçları
results = []
for col in transformations:
    stat, p = normaltest(df[col])
    results.append({'Dönüşüm Türü': col, 'p-değeri': p, 'Başarılı': 'Evet' if p >= 0.05 else 'Hayır'})

results_df = pd.DataFrame(results)
print(results_df)
