from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import os

# --- ADIM 1: TRADINGVIEW GİRİŞ BİLGİLERİNİZİ GİRİN ---
# Lütfen aşağıdaki alanları kendi TradingView kullanıcı adı ve şifrenizle doldurun.

username = os.environ.get('TRADINGVIEW_USERNAME', '')
password = os.environ.get('TRADINGVIEW_PASSWORD', '')

# Giriş yaparak TvDatafeed nesnesini oluşturun
try:
    tv = TvDatafeed(username, password)
    print("✅ TradingView'e başarıyla giriş yapıldı.")
except Exception as e:
    print(f"❌ Giriş yapılamadı: {e}")
    print("Kullanıcı adı ve şifrenizi kontrol edin. Giriş yapmadan devam edilecek.")
    tv = TvDatafeed() # Giriş başarısız olursa guest olarak devam et

# --- ADIM 2: HİSSE SENEDİ LİSTESİ VE PARAMETRELER ---
# Verileri çekilecek hisse senetlerinin sembolleri
hisse_listesi = [ 'A1CAP']

# Veri çekme parametreleri
borsa = 'BIST'
zaman_araligi = Interval.in_daily  # 1 gunluk veri
bar_sayisi = 700  # Her hisse için çekilecek veri sayısı
excel_dosya_adi = '1d_data.xlsx'

# --- ADIM 3: VERİLERİ ÇEK, KOLONLARI DÜZENLE VE BİRLEŞTİR ---
# Tüm hisse verilerini bir araya getirmek için boş bir liste oluşturun
tum_veriler = []

print("\nHisse verileri çekilmeye başlanıyor...")
for sembol in hisse_listesi:
    print(f"İşleniyor: {sembol}...", end=" ")
    try:
        # Her hisse için veriyi çek
        df = tv.get_hist(symbol=sembol, exchange=borsa, interval=zaman_araligi, n_bars=bar_sayisi)

        # Veri boş değilse, işlemlere başla
        if df is not None and not df.empty:
            # Index'i (tarih) bir sütuna dönüştür
            df_sira = df.reset_index()

            # --- DÜZELTME: İlk önce veriyi doğru şekilde yapılandır ---
            # 1. Gerekli sütunları seçerek yeni bir DataFrame oluştur
            df_yeni = df_sira[['datetime', 'high', 'low', 'close', 'volume']].copy()
            
            # 2. Hisse sembolünü içeren 'CODE' sütununu ekle. Pandas bu değeri tüm satırlara kopyalar.
            df_yeni['CODE'] = sembol

            # 3. Sütunları istenen isimlerle yeniden adlandır
            df_yeni.rename(columns={
                'datetime': 'DATE',
                'high': 'HIGH_TL',
                'low': 'LOW_TL',
                'close': 'CLOSING_TL',
                'volume': 'VOLUME_TL'
            }, inplace=True)

            # 4. Sütunları istenen sıraya getir (CODE sütunu başa gelsin)
            df_yeni = df_yeni[['CODE', 'DATE', 'HIGH_TL', 'LOW_TL', 'CLOSING_TL', 'VOLUME_TL']]
            
            # Hazırlanan veriyi listeye ekle
            tum_veriler.append(df_yeni)
            print(f"✅ Veri eklendi.")
        else:
            print(f"⚠️ Veri bulunamadı.")

    except Exception as e:
        print(f"❌ Hata oluştu: {e}")

# --- ADIM 4: TÜM VERİLERİ TEK BİR TABLODA BİRLEŞTİR VE KAYDET ---
if tum_veriler:
    print("\nVeriler birleştiriliyor...")
    # Listedeki tüm DataFrame'leri tek bir DataFrame'de birleştir
    birlesik_df = pd.concat(tum_veriler, ignore_index=True)

    try:
        # Birleştirilmiş verileri tek bir Excel sayfasına kaydet (index=False olarak)
        birlesik_df.to_excel(excel_dosya_adi, index=False, engine='openpyxl')
        print(f"🎉 İşlem tamamlandı! Tüm veriler '{excel_dosya_adi}' adlı dosyaya istenen kolon isimleriyle kaydedildi.")
    except ImportError:
        print("❌ Hata: 'openpyxl' kütüphanesi bulunamadı. Excel dosyası oluşturmak için lütfen kurun:")
        print("pip install openpyxl")
    except Exception as e:
        print(f"❌ Excel dosyası oluşturulurken genel bir hata oluştu: {e}")
else:
    print("\nHiçbir veri çekilemedi. Excel dosyası oluşturulamadı.")
