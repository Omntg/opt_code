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
        """Analiz sonuçlarını Excel formatında kaydet"""
        if self.supabase is None:
            return
            
        analysis_date = pd.to_datetime('today').strftime('%Y-%m-%d')
        
        for _, row in df.iterrows():
            # Excel formatında string değerleri koru - Excel'deki orijinal formatı kullan
            record = {
                'stock_symbol': str(row['ticker']),
                'analysis_date': analysis_date,
                
                # Signal değerleri - Excel'deki orijinal string formatını koru
                'vidya_signal': str(row.get('vidya_signal', 'Hold')),
                'vidya_bars_ago': int(row.get('vidya_bars_ago', 0)),
                'trend_signal': str(row.get('trend_signal', 'Hold')),
                'trend_bars_ago': int(row.get('trend_bars_ago', 0)),
                'linrs_signal': str(row.get('linrs_signal', 'Hold')).upper(),  # BUY/SELL formatına çevir
                'linrs_bars_ago': int(row.get('linrs_bars_ago', 0)),
                'rmt_signal': str(row.get('rmt_signal', 'Hold')),
                'rmt_bars_ago': int(row.get('rmt_bars_ago', 0)),
                'kalman_signal': str(row.get('kalman_filter_signal', 'Hold')).upper(),  # BUY/SELL formatına çevir
                'kalman_bars_ago': int(row.get('kalman_bars_ago', 0)),
                'tref_signal': str(row.get('tref_signal', 'Hold')),
                'tref_bars_ago': int(row.get('tref_bars_ago', 0)),
                
                # Filter metrikleri
                'wma_percentage_diff': float(row.get('wma_percentage_diff', 0.0)),
                'wma_filter_duration': int(row.get('wma_filter_duration', 14)),
                'lrb_percentage_diff': float(row.get('lrb_percentage_diff', 0.0)),
                'lrb_filter_duration': int(row.get('lrb_filter_duration', 14)),
                'finh_percentage_diff': float(row.get('finh_percentage_diff', 0.0)),
                'finh_filter_duration': int(row.get('finh_filter_duration', 14))
            }
            
            # Signal değerlerini Excel formatına çevir
            def convert_to_excel_format(signal_value):
                """Signal değerlerini Excel formatına çevir"""
                signal_str = str(signal_value).upper().strip()
                
                if signal_str in ['1', 'BUY', 'BUY']:
                    return 'Buy'  # Excel formatı: "Buy"
                elif signal_str in ['-1', 'SELL', 'SELL']:
                    return 'Sell'  # Excel formatı: "Sell"
                elif signal_str in ['BUY', 'BUY']:
                    return 'BUY'   # linRS ve Kalman için büyük harf
                elif signal_str in ['SELL', 'SELL']:
                    return 'SELL'  # linRS ve Kalman için büyük harf
                else:
                    return 'Hold'  # Default
                    
            # Özel signal formatları
            if 'linrs_signal' in record:
                record['linrs_signal'] = convert_to_excel_format(record['linrs_signal'])
            if 'kalman_signal' in record:
                record['kalman_signal'] = convert_to_excel_format(record['kalman_signal'])
            
            # Normal sinyaller (Buy/Sell formatı)
            for signal_col in ['vidya_signal', 'trend_signal', 'rmt_signal', 'tref_signal']:
                if signal_col in record:
                    record[signal_col] = convert_to_excel_format(record[signal_col])
            
            try:
                self.supabase.table('analysis_results').upsert(record, on_conflict='stock_symbol,analysis_date').execute()
                print(f"✅ Saved analysis for {record['stock_symbol']}")
            except Exception as e:
                print(f"❌ Error saving analysis for {record['stock_symbol']}: {e}")
    
    def save_workflow_log(self, workflow_run_id, status, message):
        """Workflow log kaydet"""
        if self.supabase is None:
            return
            
        record = {
            'workflow_run_id': workflow_run_id,
            'status': status,
            'message': message
        }
        
        try:
            self.supabase.table('workflow_logs').upsert(record, on_conflict='workflow_run_id').execute()
        except Exception as e:
            print(f"❌ Error saving workflow log: {e}")
