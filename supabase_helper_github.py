"""
Supabase Database Integration Helper - FINAL VERSION
====================================================
Trading verilerini Supabase veritabanına kaydetmek için yardımcı fonksiyonlar.
Bu dosyayı GitHub repository'nize yükleyin.
"""

import os
import pandas as pd
from datetime import datetime
import logging

# Supabase import (GitHub Actions'ta yüklenecek)
try:
    from supabase import create_client, Client
    SUPABASE_LIB_AVAILABLE = True
except ImportError:
    SUPABASE_LIB_AVAILABLE = False
    print("⚠️ Supabase Python client kurulu değil - pip install supabase gerekli")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupabaseHelper:
    def __init__(self):
        """Supabase client'ını başlat"""
        self.supabase_url = os.environ.get('SUPABASE_URL')
        self.supabase_key = os.environ.get('SUPABASE_ANON_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL ve SUPABASE_ANON_KEY environment değişkenleri gerekli")
        
        if not SUPABASE_LIB_AVAILABLE:
            raise ImportError("Supabase Python client kurulu değil")
        
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info("✅ Supabase client başarıyla oluşturuldu")
        except Exception as e:
            logger.error(f"❌ Supabase client oluşturulurken hata: {e}")
            raise

    def save_raw_stock_data(self, df):
        """
        Ham hisse senedi verilerini raw_stock_data tablosuna kaydet
        
        Args:
            df: Hisse senedi verileri DataFrame'i (tv_data.py çıktısı)
        """
        try:
            # Veri hazırlama
            records = []
            for _, row in df.iterrows():
                record = {
                    'code': str(row['CODE']),
                    'date': pd.to_datetime(row['DATE']).strftime('%Y-%m-%d'),
                    'high_tl': float(row['HIGH_TL']),
                    'low_tl': float(row['LOW_TL']),
                    'closing_tl': float(row['CLOSING_TL']),
                    'volume_tl': int(row['VOLUME_TL'])
                }
                records.append(record)
            
            if not records:
                logger.warning("⚠️ Kaydedilecek ham veri bulunamadı")
                return False
            
            # Toplu kayıt (upsert ile)
            result = self.supabase.table('raw_stock_data').upsert(
                records,
                on_conflict='code,date'
            ).execute()
            
            logger.info(f"✅ {len(records)} ham veri kaydı başarıyla kaydedildi")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ham veriler kaydedilirken hata: {e}")
            return False

    def save_analysis_results(self, results_df):
        """
        Analiz sonuçlarını analysis_results tablosuna kaydet
        
        Args:
            results_df: Analiz sonuçları DataFrame'i (analiz.py çıktısı)
        """
        try:
            if results_df.empty:
                logger.warning("⚠️ Kaydedilecek analiz sonucu bulunamadı")
                return False
            
            # Veri hazırlama - gerçek Supabase kolon adlarını kullan
            records = []
            for _, row in results_df.iterrows():
                record = {
                    'stock_symbol': str(row.get('ticker', row.get('stock_symbol', ''))),
                    'analysis_date': pd.to_datetime(row['date']).strftime('%Y-%m-%d'),
                    'wma_percentage_diff': float(row['wma_percentage_diff']) if pd.notna(row['wma_percentage_diff']) else None,
                    'wma_filter_duration': int(row['wma_filter_duration']) if pd.notna(row['wma_filter_duration']) else None,
                    'lrb_percentage_diff': float(row['lrb_percentage_diff']) if pd.notna(row['lrb_percentage_diff']) else None,
                    'lrb_filter_duration': int(row['lrb_filter_duration']) if pd.notna(row['lrb_filter_duration']) else None,
                    'finh_percentage_diff': float(row['finh_percentage_diff']) if pd.notna(row['finh_percentage_diff']) else None,
                    'finh_filter_duration': int(row['finh_filter_duration']) if pd.notna(row['finh_filter_duration']) else None,
                    'vidya_signal': int(row.get('vidya_signal', 0)),
                    'trend_signal': int(row.get('trend_signal', 0)),
                    'linrs_signal': int(row.get('linrs_signal', 0)),
                    'rmt_signal': int(row.get('rmt_signal', 0)),
                    'kalman_signal': int(row.get('kalman_filter_signal', row.get('kalman_signal', 0))),
                    'tref_signal': int(row.get('tref_signal', 0))
                }
                records.append(record)
            
            # Toplu kayıt (upsert ile)
            result = self.supabase.table('analysis_results').upsert(
                records,
                on_conflict='stock_symbol,analysis_date'
            ).execute()
            
            logger.info(f"✅ {len(records)} analiz sonucu başarıyla kaydedildi")
            return True
            
        except Exception as e:
            logger.error(f"❌ Analiz sonuçları kaydedilirken hata: {e}")
            return False

    def save_workflow_log(self, workflow_run_id, status, execution_time_seconds=None, 
                         stocks_processed=None, analysis_completed=None, files_created=None, error_message=None):
        """
        Workflow çalışma loglarını workflow_logs tablosuna kaydet
        
        Args:
            workflow_run_id: GitHub Actions run ID
            status: 'success' veya 'failure'
            execution_time_seconds: Çalışma süresi (saniye)
            stocks_processed: İşlenen hisse sayısı
            analysis_completed: Analiz tamamlanan sayısı
            files_created: Oluşturulan dosyalar listesi
            error_message: Hata mesajı
        """
        try:
            log_record = {
                'workflow_run_id': workflow_run_id,
                'status': status,
                'execution_time_seconds': execution_time_seconds,
                'stocks_processed': stocks_processed,
                'analysis_completed': analysis_completed,
                'files_created': files_created,
                'error_message': error_message
            }
            
            # None değerleri temizle
            log_record = {k: v for k, v in log_record.items() if v is not None}
            
            # INSERT kullan (Upsert değil)
            result = self.supabase.table('workflow_logs').insert(log_record).execute()
            logger.info(f"✅ Workflow log başarıyla kaydedildi: {status}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Workflow log kaydedilirken hata: {e}")
            return False

    def test_connection(self):
        """Supabase bağlantısını test et"""
        try:
            # Basit bir sorgu ile bağlantıyı test et
            result = self.supabase.table('raw_stock_data').select('id').limit(1).execute()
            logger.info("✅ Supabase bağlantısı başarılı")
            return True
        except Exception as e:
            logger.error(f"❌ Supabase bağlantısı başarısız: {e}")
            return False


def main():
    """Test fonksiyonu"""
    try:
        helper = SupabaseHelper()
        helper.test_connection()
    except Exception as e:
        print(f"Test başarısız: {e}")


if __name__ == "__main__":
    main()
