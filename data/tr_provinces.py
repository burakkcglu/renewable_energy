import pandas as pd
from geopy.geocoders import Nominatim
import time

# 81 İl Listesi
provinces = [
    "Adana", "Adiyaman", "Afyonkarahisar", "Agri", "Amasya", "Ankara", "Antalya", "Artvin", "Aydin", "Balikesir",
    "Bilecik", "Bingol", "Bitlis", "Bolu", "Burdur", "Bursa", "Canakkale", "Cankiri", "Corum", "Denizli",
    "Diyarbakir", "Edirne", "Elazig", "Erzincan", "Erzurum", "Eskisehir", "Gaziantep", "Giresun", "Gumushane", "Hakkari",
    "Hatay", "Isparta", "Mersin", "Istanbul", "Izmir", "Kars", "Kastamonu", "Kayseri", "Kirklareli", "Kirsehir",
    "Kocaeli", "Konya", "Kutahya", "Malatya", "Manisa", "Kahramanmaras", "Mardin", "Mugla", "Mus", "Nevsehir",
    "Nigde", "Ordu", "Rize", "Sakarya", "Samsun", "Siirt", "Sinop", "Sivas", "Tekirdag", "Tokat",
    "Trabzon", "Tunceli", "Sanliurfa", "Usak", "Van", "Yozgat", "Zonguldak", "Aksaray", "Bayburt", "Karaman",
    "Kirikkale", "Batman", "Sirnak", "Bartin", "Ardahan", "Igdir", "Yalova", "Karabuk", "Kilis", "Osmaniye", "Duzce"
]

geolocator = Nominatim(user_agent="renewable_portfolio_tr")
coord_list = []

print("İl koordinatları çekiliyor... (Bu işlem yaklașık 1-2 dakika sürebilir)")
for prov in provinces:
    try:
        # Türkiye sınırlarında arama yapması için "Turkey" kelimesini ekliyoruz
        location = geolocator.geocode(f"{prov}, Turkey")
        if location:
            coord_list.append({
                "province": prov,
                "lat": location.latitude,
                "lon": location.longitude
            })
        else:
            print(f"Hata: {prov} bulunamadı.")
        time.sleep(1) # API'yi yormamak ve banlanmamak için 1 saniye bekletiyoruz
    except Exception as e:
        print(f"{prov} çekilirken hata oluştu: {e}")

# Dataframe oluşturma ve kaydetme
df_coords = pd.DataFrame(coord_list)
df_coords.to_csv("tr_provinces_coords.csv", index=False)
print("Koordinat dosyası 'tr_provinces_coords.csv' olarak başarıyla kaydedildi!")
print(df_coords.head())