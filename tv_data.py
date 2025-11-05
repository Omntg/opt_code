from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import os
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ADIM 1: TRADINGVIEW GİRİŞ BİLGİLERİNİZİ GİRİN ---
username = os.environ.get('TRADINGVIEW_USERNAME', '')
password = os.environ.get('TRADINGVIEW_PASSWORD', '')

print("🔐 TradingView giriş bilgileri kontrol ediliyor...")

# Eğer kullanıcı adı veya şifre eksikse uyarı ver
if not username or not password:
    print("⚠️ TRADINGVIEW_USERNAME veya TRADINGVIEW_PASSWORD ortam değişkenleri eksik!")
    print("GitHub Repository'nizde → Settings → Secrets and variables → Actions")
    print("TRADINGVIEW_USERNAME ve TRADINGVIEW_PASSWORD ekleyin.")
    print("Kullanıcı adı: TradingView'e giriş yaptığınız email/kullanıcı adı")
    print("Şifre: TradingView şifreniz")
    print("\n❌ Guest modda çalışılacak - veri erişimi sınırlı olacak!")
    tv = TvDatafeed()  # Guest olarak devam et
    login_success = False
else:
    print(f"📧 Kullanıcı adı bulundu: {username[:3]}***")
    try:
        tv = TvDatafeed(username, password)
        login_success = True
        print("✅ TradingView'e başarıyla giriş yapıldı!")
    except Exception as e:
        print(f"❌ Giriş yapılamadı: {e}")
        print("⚠️ Guest modda çalışılacak - veri erişimi sınırlı olacak!")
        tv = TvDatafeed()  # Giriş başarısız olursa guest olarak devam et
        login_success = False

# --- ADIM 2: HİSSE SENEDİ LİSTESİ VE PARAMETRELER ---
hisse_listesi = [ 'A1CAP', 'A1YEN', 'AEFES', 'AGESA', 'AGHOL', 'AGYO', 'AHGAZ', 'AKBNK', 'AKFGY', 'AKGRT', 'AKMGY', 'AKSEN', 'AKSUE', 'ALBRK', 'ALCAR', 'ALKA', 'ALTIN', 'ANHYT', 'ANSGR', 'ARASE', 'ARDYZ', 'ASELS', 'ASTOR', 'ATAGY', 'ATATP', 'AVGYO', 'AYDEM', 'AYEN', 'AYGAZ', 'BAGFS', 'BAKAB', 'BASGZ', 'BESLR', 'BEYAZ', 'BIGCH', 'BIMAS', 'BNTAS', 'BOSSA', 'BRKSN', 'BRLSM', 'BRSAN', 'BRYAT', 'CCOLA', 'CEMTS', 'CIMSA', 'CLEBI', 'CRDFA', 'CWENE', 'DAPGM', 'DERIM', 'DESA', 'DESPC', 'DGATE', 'DOCO', 'DOFER', 'DOHOL', 'EBEBK', 'ECZYT', 'EDATA', 'EGEPO', 'EGGUB', 'EGPRO', 'EKGYO', 'ELITE', 'EMKEL', 'ENERY', 'ENJSA', 'ENKAI', 'EREGL', 'EUPWR', 'EUREN', 'FMIZP', 'FORTE', 'FROTO', 'FZLGY', 'GARAN', 'GARFA', 'GEDZA', 'GENIL', 'GENTS', 'GESAN', 'GIPTA', 'GLCVY', 'GLDTR', 'GLRMK', 'GLYHO', 'GMSTR', 'GMTAS', 'GOKNR', 'GRSEL', 'GRTHO', 'GUBRF', 'GWIND', 'HALKB', 'HLGYO', 'HTTBT', 'HUNER', 'INDES', 'ISCTR', 'ISDMR', 'ISFIN', 'ISGSY', 'ISGYO', 'ISKPL', 'ISMEN', 'KATMR', 'KCAER', 'KCHOL', 'KLKIM', 'KLMSN', 'KLSYN', 'KOZAA', 'KOZAL', 'KRDMA', 'KRDMD', 'KRONT', 'KRPLS', 'KRSTL', 'LIDER', 'LIDFA', 'LILAK', 'LINK', 'LKMNH', 'LOGO', 'LYDYE', 'MACKO', 'MAGEN', 'MAKTK', 'MARBL', 'MAVI', 'MERIT', 'METUR', 'MGROS', 'MIATK', 'MNDRS', 'MOBTL', 'MPARK', 'MRGYO', 'MTRKS', 'NTGAZ', 'NTHOL', 'NUHCM', 'OBASE', 'ODAS', 'OFSYM', 'ONCSM', 'ORGE', 'OTKAR', 'OYAKC', 'OYYAT', 'OZGYO', 'OZSUB', 'PAGYO', 'PAPIL', 'PASEU', 'PATEK', 'PETUN', 'PGSUS', 'PINSU', 'PLTUR', 'PNLSN', 'PRKME', 'PSDTC', 'QUAGR', 'RNPOL', 'RYGYO', 'RYSAS', 'SAHOL', 'SANEL', 'SAYAS', 'SDTTR', 'SELGD', 'SISE', 'SKBNK', 'SMART', 'SRVGY', 'SUNTK', 'SUWEN', 'TABGD', 'TARKM', 'TATGD', 'TAVHL', 'TBORG', 'TCELL', 'TEZOL', 'THYAO', 'TLMAN', 'TMPOL', 'TNZTP', 'TRCAS', 'TRGYO', 'TSKB', 'TTKOM', 'TUKAS', 'TUPRS', 'TURSG', 'ULKER', 'ULUUN', 'VAKBN', 'VERUS', 'YGGYO', 'YKBNK', 'YUNSA', 'YYLGD', 'ZRGYO'
    ]  

borsa = 'BIST'
zaman_araligi = Interval.in_daily  # 1 gunluk veri
bar_sayisi = 700  # Her hisse için çekilecek veri sayısı
excel_dosya_adi = '1d_data.xlsx'

# --- ADIM 3: VERİLERİ ÇEK, KOLONLARI DÜZENLE VE BİRLEŞTİR ---
tum_veriler = []

print(f"\n📊 Hisse verileri çekilmeye başlanıyor...")
print(f"🔐 Giriş durumu: {'✅ Giriş yapıldı' if login_success else '❌ Guest mod (sınırlı veri)'}")

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
            print("✅")
        else:
            print("⚠️")

    except Exception as e:
        print(f"❌")

# --- ADIM 4: TÜM VERİLERİ TEK BİR TABLODA BİRLEŞTİR VE KAYDET ---
if tum_veriler:
    print(f"\n📈 Veriler birleştiriliyor...")
    # Listedeki tüm DataFrame'leri tek bir DataFrame'de birleştir
    birlesik_df = pd.concat(tum_veriler, ignore_index=True)

    try:
        # Birleştirilmiş verileri tek bir Excel sayfasına kaydet (index=False olarak)
        birlesik_df.to_excel(excel_dosya_adi, index=False, engine='openpyxl')
        print(f"🎉 İşlem tamamlandı! Tüm veriler '{excel_dosya_adi}' adlı dosyaya istenen kolon isimleriyle kaydedildi.")
        print(f"📊 Toplam {len(tum_veriler)} hisse için veri çekildi.")
        print(f"🔗 Giriş durumu: {'Giriş yapıldı' if login_success else 'Guest mod (sınırlı veri erişimi)'}")
    except ImportError:
        print("❌ Hata: 'openpyxl' kütüphanesi bulunamadı. Excel dosyası oluşturmak için lütfen kurun:")
        print("pip install openpyxl")
    except Exception as e:
        print(f"❌ Excel dosyası oluşturulurken genel bir hata oluştu: {e}")
else:
    print("\n❌ Hiçbir veri çekilemedi. Excel dosyası oluşturulamadı.")
