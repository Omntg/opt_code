"""
Supabase Database Integration Helper
====================================
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
            
            # Tarihi belirle (bugün)
            analysis_date = datetime.now().strftime('%Y-%m-%d')
            
            # Veri hazırlama
            records = []
            for _, row in results_df.iterrows():
                record = {
                    'stock_symbol': str(row['Stock Symbol']),
                    'analysis_date': analysis_date,
                    'vidya_signal': str(row['VIDYA Signal']) if pd.notna(row['VIDYA Signal']) else None,
                    'vidya_bars_ago': int(row['VIDYA Signal – Bars Ago']) if pd.notna(row['VIDYA Signal – Bars Ago']) else None,
                    'trend_signal': str(row['Trend Signal']) if pd.notna(row['Trend Signal']) else None,
                    'trend_bars_ago': int(row['Trend Signal – Bars Ago']) if pd.notna(row['Trend Signal – Bars Ago']) else None,
                    'linrs_signal': str(row['linRS Signal']) if pd.notna(row['linRS Signal']) else None,
                    'linrs_bars_ago': int(row['linRS Signal – Bars Ago']) if pd.notna(row['linRS Signal – Bars Ago']) else None,
                    'rmt_signal': str(row['RMT Signal']) if pd.notna(row['RMT Signal']) else None,
                    'rmt_bars_ago': int(row['RMT Signal – Bars Ago']) if pd.notna(row['RMT Signal – Bars Ago']) else None,
                    'kalman_signal': str(row['Kalman Filter Signal']) if pd.notna(row['Kalman Filter Signal']) else None,
                    'kalman_bars_ago': int(row['Kalman Signal – Bars Ago']) if pd.notna(row['Kalman Signal – Bars Ago']) else None,
                    'tref_signal': str(row['TREF Signal']) if pd.notna(row['TREF Signal']) else None,
                    'tref_bars_ago': int(row['TREF Signal – Bars Ago']) if pd.notna(row['TREF Signal – Bars Ago']) else None,
                    'wma_percentage_diff': float(row['WMA Percentage Diff']) if pd.notna(row['WMA Percentage Diff']) else None,
                    'wma_filter_duration': int(row['WMA Filter Duration']) if pd.notna(row['WMA Filter Duration']) else None,
                    'lrb_percentage_diff': float(row['LRB Percentage Diff']) if pd.notna(row['LRB Percentage Diff']) else None,
                    'lrb_filter_duration': int(row['LRB Filter Duration']) if pd.notna(row['LRB Filter Duration']) else None,
                    'finh_percentage_diff': float(row['FINH Percentage Diff']) if pd.notna(row['FINH Percentage Diff']) else None,
                    'finh_filter_duration': int(row['FINH Filter Duration']) if pd.notna(row['FINH Filter Duration']) else None
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
                         stocks_processed=None, analysis_completed=False, error_message=None,
                         files_created=None, git_commit_hash=None):
        """
        Workflow çalışma loglarını workflow_logs tablosuna kaydet
        
        Args:
            workflow_run_id: GitHub Actions run ID
            status: 'success' veya 'failure'
            execution_time_seconds: Çalışma süresi (saniye)
            stocks_processed: İşlenen hisse sayısı
            analysis_completed: Analiz tamamlandı mı
            error_message: Hata mesajı
            files_created: Oluşturulan dosyalar
            git_commit_hash: Git commit hash
        """
        try:
            log_record = {
                'workflow_run_id': workflow_run_id,
                'status': status,
                'execution_time_seconds': execution_time_seconds,
                'stocks_processed': stocks_processed,
                'analysis_completed': analysis_completed,
                'error_message': error_message,
                'files_created': files_created,
                'git_commit_hash': git_commit_hash
            }
            
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
