import pandas as pd
from supabase import create_client, Client
import os

class SupabaseHelper:
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            print("⚠️ Supabase credentials not found - skipping database operations")
            self.supabase = None
            return
            
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            print("✅ Supabase connection successful")
        except Exception as e:
            print(f"❌ Supabase connection failed: {e}")
            self.supabase = None
    
    def test_connection(self):
        """Supabase bağlantısını test et"""
        if self.supabase is None:
            return False
        try:
            # Basit bir sorgu ile bağlantıyı test et
            result = self.supabase.table('raw_stock_data').select('count').limit(1).execute()
            return True
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def save_raw_stock_data(self, df):
        """Ham hisse senedi verilerini kaydet"""
        if self.supabase is None:
            return
            
        for _, row in df.iterrows():
            record = {
                'code': str(row['CODE']),
                'date': pd.to_datetime(row['DATE']).strftime('%Y-%m-%d'),
                'high_tl': float(row['HIGH_TL']),
                'low_tl': float(row['LOW_TL']),
                'closing_tl': float(row['CLOSING_TL']),
                'volume_tl': int(row['VOLUME_TL'])
            }
            
            try:
                self.supabase.table('raw_stock_data').upsert(record, on_conflict='code,date').execute()
            except Exception as e:
                print(f"❌ Error saving raw stock data for {record['code']}: {e}")
    
    def save_analysis_results(self, df):
        """Analiz sonuçlarını Excel'den otomatik olarak kaydet"""
        if self.supabase is None:
            return
            
        analysis_date = pd.to_datetime('today').strftime('%Y-%m-%d')
        
        for _, row in df.iterrows():
            # Excel'deki doğru kolon adlarını kullan
            record = {
                'stock_symbol': str(row['Stock Symbol']),  # Excel kolon adı
                'analysis_date': analysis_date,
                
                # VIDYA analizi
                'vidya_signal': str(row['VIDYA Signal']),
                'vidya_bars_ago': int(row['VIDYA Signal – Bars Ago']),
                
                # Trend analizi  
                'trend_signal': str(row['Trend Signal']),
                'trend_bars_ago': int(row['Trend Signal – Bars Ago']),
                
                # linRS analizi
                'linrs_signal': str(row['linRS Signal']).upper(),
                'linrs_bars_ago': int(row['linRS Signal – Bars Ago']),
                
                # RMT analizi
                'rmt_signal': str(row['RMT Signal']),
                'rmt_bars_ago': int(row['RMT Signal – Bars Ago']),
                
                # Kalman analizi
                'kalman_signal': str(row['Kalman Filter Signal']).upper(),
                'kalman_bars_ago': int(row['Kalman Signal – Bars Ago']),
                
                # TREF analizi
                'tref_signal': str(row['TREF Signal']),
                'tref_bars_ago': int(row['TREF Signal – Bars Ago']),
                
                # WMA metrikleri
                'wma_percentage_diff': float(row['WMA Percentage Diff']),
                'wma_filter_duration': int(row['WMA Filter Duration']),
                
                # LRB metrikleri
                'lrb_percentage_diff': float(row['LRB Percentage Diff']),
                'lrb_filter_duration': int(row['LRB Filter Duration']),
                
                # FINH metrikleri
                'finh_percentage_diff': float(row['FINH Percentage Diff']),
                'finh_filter_duration': int(row['FINH Filter Duration'])
            }
            
            # Signal formatlarını Excel formatına çevir
            def convert_to_excel_format(signal_value):
                signal_str = str(signal_value).upper().strip()
                
                if signal_str in ['1', 'BUY']:
                    return 'Buy'  # Normal Buy formatı
                elif signal_str in ['-1', 'SELL']:
                    return 'Sell'  # Normal Sell formatı
                elif signal_str == 'BUY':
                    return 'BUY'   # Büyük harf formatı (linRS/Kalman için)
                elif signal_str == 'SELL':
                    return 'SELL'  # Büyük harf formatı (linRS/Kalman için)
                else:
                    return 'Hold'  # Hold formatı
                
            # Tüm sinyalleri dönüştür
            record['vidya_signal'] = convert_to_excel_format(record['vidya_signal'])
            record['trend_signal'] = convert_to_excel_format(record['trend_signal'])
            record['linrs_signal'] = convert_to_excel_format(record['linrs_signal'])
            record['rmt_signal'] = convert_to_excel_format(record['rmt_signal'])
            record['kalman_signal'] = convert_to_excel_format(record['kalman_signal'])
            record['tref_signal'] = convert_to_excel_format(record['tref_signal'])
            
            try:
                self.supabase.table('analysis_results').upsert(record, on_conflict='stock_symbol,analysis_date').execute()
                print(f"✅ Saved analysis for {record['stock_symbol']}")
            except Exception as e:
                print(f"❌ Error saving analysis for {record['stock_symbol']}: {e}")
    
    def save_workflow_log(self, workflow_run_id, status, execution_time_seconds, stocks_processed, analysis_completed, files_created, message=None):
        """Workflow log kaydet"""
        if self.supabase is None:
            return
            
        record = {
            'workflow_run_id': workflow_run_id,
            'status': status,
            'execution_time_seconds': execution_time_seconds,
            'stocks_processed': stocks_processed,
            'analysis_completed': analysis_completed,
            'files_created': files_created
        }
        
        if message:
            record['message'] = message
        
        try:
            self.supabase.table('workflow_logs').upsert(record, on_conflict='workflow_run_id').execute()
            print(f"✅ Workflow log saved: {workflow_run_id}")
        except Exception as e:
            print(f"❌ Error saving workflow log: {e}")
