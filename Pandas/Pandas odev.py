import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#burada veri setimizi yükleyip ilk 5 satırı gösteren kodu yazacağız

df = pd.read_csv("C:/Users/enesa/OneDrive/Desktop/Ders/house_prices.csv")

print(df.head().to_string())


#Burada gereksiz stunları kaldırıp tamamen boş olan satırları temizleyeceğiz
df.dropna(axis=1,how='all',inplace=True)
print(df.head())
print(df.isnull().sum())


# Stunlardaki eksik değerlerin sayısını kontrol etme
 print(df.isnull().sum())



# burada is Price stunlarındaki boş değerleri ortalama ile dolduruyoruz
df["Price (in rupees)"] = df["Price (in rupees)"].fillna(df["Price (in rupees)"].mean())
print(df["Price (in rupees)"])




#Şimdi şehirlerdeki lokasyonları gruplayarak ortalama ev fiyalatlarını hesaplayalım
city_avg_price = df.groupby("location")["Price (in rupees)"].mean().sort_values(ascending=False)
print(city_avg_price)




# Lokasyon bazında ortalama fiyatları hesaplıyoruz
    df["Price (in rupees)"] = pd.to_numeric(df["Price (in rupees)"], errors="coerce")  # Fiyatları sayısal forma çeviriyoruz
    avg_prices = df.groupby("location")["Price (in rupees)"].mean().sort_values(ascending=False).head(25)  # İlk 25 lokasyonu al

    # Grafiği çizelim
    plt.figure(figsize=(12, 6))
    avg_prices.plot(kind="bar", color="skyblue")
    plt.xlabel("Lokasyon")
    plt.ylabel("Ortalama Fiyat")
    plt.title("Lokasyonlara Göre Ortalama Ev Fiyatları")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()









