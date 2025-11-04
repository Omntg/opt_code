import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# YARDIMCI FONKSİYONLAR
# ---------------------------

def wma(data, period):
    """
    Belirtilen periyot için ağırlıklı hareketli ortalama (WMA) hesaplar.
    """
    weights = np.arange(1, period + 1)
    return data.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def rolling_linreg(series, period):
    """
    Verilen seriye (örn. HLC3) göre, sabit x değerleri (0, 1, ..., period-1) üzerinden
    rolling pencere uygulayarak lineer regresyon sonucu (pencerenin sonundaki tahmin)
    ve eğimi hesaplar.
    """
    reg_vals = pd.Series(index=series.index, dtype='float64')
    slopes = pd.Series(index=series.index, dtype='float64')
    x = np.arange(period)
    x_mean = x.mean()
    denom = np.sum((x - x_mean) ** 2)
    
    for i in range(period - 1, len(series)):
        y = series.iloc[i - period + 1 : i + 1].values
        y_mean = y.mean()
        slope = np.sum((x - x_mean) * (y - y_mean)) / denom
        intercept = y_mean - slope * x_mean
        reg_val = intercept + slope * (period - 1)
        reg_vals.iloc[i] = reg_val
        slopes.iloc[i] = slope
        
    return reg_vals, slopes

def count_consecutive(df, col, target_value):
    """
    DataFrame'in sonundan başlayarak, belirli sütunda target_value'nun kaç ardışık satırda yer aldığını sayar.
    """
    count = 0
    for val in df[col].iloc[::-1]:
        if val == target_value:
            count += 1
        else:
            break
    return count

def finh_ema(series, period):
    """
    FINH_ema fonksiyonu: x serisine, period periyodu için üstel hareketli ortalama (EMA)
    hesaplar. İlk değer için EMA[0] = alpha * x[0] olarak alınır.
    """
    alpha = 2 / (period + 1)
    result = pd.Series(index=series.index, dtype='float64')
    for i in range(len(series)):
        if i == 0:
            result.iloc[i] = alpha * series.iloc[i]
        else:
            result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]
    return result

# ---------------------------
# ANALİZ FONKSİYONLARI
# ---------------------------

def analyze_stock_wma(df, ticker, n=146):
    """
    WMA tabanlı analiz:
      - CLOSING_TL üzerinden iki farklı WMA hesaplanır (n ve n/2 periyotlu),
      - İkisi arasındaki farkın WMA'sı (mab) alınır,
      - Günlük filtre: CLOSING_TL > mab ise "green", < ise "red", eşitse "neutral",
      - En son gün için yüzde fark ve ardışık gün sayısı hesaplanır.
    
    Çıktı sözlüğü:
      • Ticker
      • WMA_Percentage Diff
      • WMA_Filter Duration
    """
    df = df.sort_values(by='DATE')
    
    period_half = round(n / 2)
    n2ma = 2 * wma(df['CLOSING_TL'], period_half)
    nma = wma(df['CLOSING_TL'], n)
    diff = n2ma - nma
    sqn = round(np.sqrt(n))
    mab = wma(diff, sqn)
    
    df['mab'] = mab
    if df['mab'].dropna().empty:
        print(f"{ticker} için yeterli veri yok (WMA analizi).")
        return None

    df['WMA_Filter'] = np.where(df['CLOSING_TL'] > df['mab'], 'green',
                                 np.where(df['CLOSING_TL'] < df['mab'], 'red', 'neutral'))
    
    latest_close = df['CLOSING_TL'].iloc[-1]
    latest_mab = df['mab'].iloc[-1]
    
    pct_diff = ((latest_close - latest_mab) / latest_mab) * 100 if latest_mab != 0 else np.nan
    latest_filter = 'green' if latest_close > latest_mab else ('red' if latest_close < latest_mab else 'neutral')
    filter_duration = count_consecutive(df, 'WMA_Filter', latest_filter)
    
    return {
        'Ticker': ticker,
        'WMA_Percentage Diff': pct_diff,
        'WMA_Filter Duration': filter_duration
    }

def analyze_stock_linreg(df, ticker, period=105):
    """
    Rolling lineer regresyon (LRB) tabanlı analiz:
      - HLC3 değeri hesaplanır: (CLOSING_TL + LOW_TL + HIGH_TL) / 3,
      - Rolling pencere ile lineer regresyon sonucu (Reg) ve eğim (Slope) hesaplanır,
      - Kapanış değeri ile regresyon sonucu arasındaki yüzde fark ve 
        son durumun ardışık gün sayısı hesaplanır.
        
    Çıktı sözlüğü:
      • Ticker
      • LRB_Percentage Diff
      • LRB_Filter Duration
    """
    df = df.sort_values(by='DATE')
    df['HLC3'] = (df['CLOSING_TL'] + df['LOW_TL'] + df['HIGH_TL']) / 3
    
    reg_vals, slopes = rolling_linreg(df['HLC3'], period)
    df['Reg'] = reg_vals
    df['Slope'] = slopes
    df = df.dropna(subset=['Reg']).copy()
    
    if df.empty:
        print(f"{ticker} için yeterli veri bulunamadı (LRB analizi).")
        return None
    
    last_row = df.iloc[-1]
    last_close = last_row['CLOSING_TL']
    last_reg = last_row['Reg']
    
    pct_diff = ((last_close - last_reg) / last_reg) * 100 if last_reg != 0 else np.nan
    
    df['LRB_Filter'] = np.where(df['CLOSING_TL'] > df['Reg'], 'green',
                                 np.where(df['CLOSING_TL'] < df['Reg'], 'red', 'neutral'))
    last_filter = 'green' if last_close > last_reg else ('red' if last_close < last_reg else 'neutral')
    filter_duration = count_consecutive(df, 'LRB_Filter', last_filter)
    
    return {
        'Ticker': ticker,
        'LRB_Percentage Diff': pct_diff,
        'LRB_Filter Duration': filter_duration
    }

def analyze_stock_finh(df, ticker, period=110, yukdus=True):
    """
    FINH filtresi analizi (PineScript kodunun Python'a uyarlanması):
      - CLOSING_TL serisi üzerinden, önce period ve period/2 ile EMA hesaplanır,
        sonra FINH = FINH_ema(2*EMA(period/2) - EMA(period), sqrt(period)) formülü uygulanır.
      - Filtre: CLOSING_TL > FINH → green, < → red, eşitse neutral,
      - En son gün için kapanış ile FINH arasındaki yüzde fark ve ardışık gün sayısı hesaplanır.
      
    Çıktı sözlüğü:
      • Ticker
      • FINH_Percentage Diff
      • FINH_Filter Duration
    """
    df = df.sort_values(by='DATE')
    sqrtPeriod = np.sqrt(period)
    
    ema_half = finh_ema(df['CLOSING_TL'], period / 2)
    ema_full = finh_ema(df['CLOSING_TL'], period)
    intermediate = 2 * ema_half - ema_full
    FINH = finh_ema(intermediate, sqrtPeriod)
    df['FINH'] = FINH

    df['FINH_Filter'] = np.where(df['CLOSING_TL'] > df['FINH'], 'green',
                                 np.where(df['CLOSING_TL'] < df['FINH'], 'red', 'neutral'))
    
    if df['FINH'].dropna().empty:
        print(f"{ticker} için yeterli veri yok (FINH analizi).")
        return None

    latest_close = df['CLOSING_TL'].iloc[-1]
    latest_finh = df['FINH'].iloc[-1]
    
    pct_diff = ((latest_close - latest_finh) / latest_finh) * 100 if latest_finh != 0 else np.nan
    latest_filter = 'green' if latest_close > latest_finh else ('red' if latest_close < latest_finh else 'neutral')
    filter_duration = count_consecutive(df, 'FINH_Filter', latest_filter)
    
    return {
        'Ticker': ticker,
        'FINH_Percentage Diff': pct_diff,
        'FINH_Filter Duration': filter_duration
    }

# ---------------------------
# VIDYA STRATEGY
# ---------------------------

class VIDYAStrategy:
    def __init__(self, len_V=2, len_VHist=5, strat_loop=1, end_loop=60, long_t=40, short_t=10):
        self.len_V = len_V
        self.len_VHist = len_VHist
        self.strat_loop = strat_loop
        self.end_loop = end_loop
        self.long_t = long_t
        self.short_t = short_t
    
    def calculate_vidya(self, close_prices):
        try:
            stdev1 = close_prices.rolling(window=self.len_V).std()
            stdev2 = close_prices.rolling(window=self.len_VHist).std()
            
            with np.errstate(divide='ignore', invalid='ignore'):
                SD_V = stdev1 / stdev2
            SD_V = SD_V.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            SC = 2.0 / (self.len_V + 1)
            vidya = pd.Series(index=close_prices.index, dtype='float64')
            vidya.iloc[0] = close_prices.iloc[0]
            
            for i in range(1, len(close_prices)):
                if pd.isna(SD_V.iloc[i]):
                    vidya.iloc[i] = SC * close_prices.iloc[i] + (1 - SC) * vidya.iloc[i-1]
                else:
                    vidya.iloc[i] = SD_V.iloc[i] * SC * close_prices.iloc[i] + (1 - SD_V.iloc[i] * SC) * vidya.iloc[i-1]
            
            return vidya
        except Exception as e:
            print(f"Error in calculate_vidya: {e}")
            return pd.Series(np.full(len(close_prices), np.nan), index=close_prices.index)
    
    def calculate_loop_f(self, vidya_values):
        try:
            loop_values = pd.Series(index=vidya_values.index, dtype='float64')
            
            for i in range(len(vidya_values)):
                sum_val = 0.0
                for j in range(self.strat_loop, self.end_loop + 1):
                    if i - j >= 0:
                        if vidya_values.iloc[i] > vidya_values.iloc[i - j]:
                            sum_val += 1
                        else:
                            sum_val -= 1
                loop_values.iloc[i] = sum_val
            
            return loop_values
        except Exception as e:
            print(f"Error in calculate_loop_f: {e}")
            return pd.Series(np.full(len(vidya_values), np.nan), index=vidya_values.index)
    
    def generate_signals(self, close_prices):
        try:
            vidya_values = self.calculate_vidya(close_prices)
            loop_values = self.calculate_loop_f(vidya_values)
            signals = pd.Series(0, index=close_prices.index)
            
            for i in range(len(signals)):
                long_condition = loop_values.iloc[i] > self.long_t
                short_condition = loop_values.iloc[i] < self.short_t
                
                if i == 0:
                    if long_condition:
                        signals.iloc[i] = 1
                    elif short_condition:
                        signals.iloc[i] = -1
                    else:
                        signals.iloc[i] = 0
                else:
                    if long_condition and not short_condition:
                        signals.iloc[i] = 1
                    elif short_condition:
                        signals.iloc[i] = -1
                    else:
                        signals.iloc[i] = signals.iloc[i-1]
            
            return signals, vidya_values, loop_values
        except Exception as e:
            print(f"Error in generate_signals: {e}")
            return pd.Series(0, index=close_prices.index), pd.Series(np.full(len(close_prices), np.nan), index=close_prices.index), pd.Series(np.full(len(close_prices), np.nan), index=close_prices.index)
    
    def get_last_signal(self, close_prices):
        try:
            signals, vidya_values, loop_values = self.generate_signals(close_prices)
            
            if vidya_values.isna().all():
                print("Warning: VIDYA calculation failed, all values are NaN")
                return None, None
            
            last_signal_type = None
            last_signal_candle_distance = 0
            
            for i in range(len(signals)-1, 0, -1):
                if signals.iloc[i] != signals.iloc[i-1]:
                    last_signal_candle_distance = len(signals) - 1 - i
                    if signals.iloc[i] == 1:
                        last_signal_type = "Buy"
                    elif signals.iloc[i] == -1:
                        last_signal_type = "Sell"
                    break
            
            if last_signal_type is None and len(signals) > 0:
                if signals.iloc[-1] == 1:
                    last_signal_type = "Buy"
                elif signals.iloc[-1] == -1:
                    last_signal_type = "Sell"
                last_signal_candle_distance = 0
            
            return last_signal_type, last_signal_candle_distance
        except Exception as e:
            print(f"Error in get_last_signal: {e}")
            return None, None

# ---------------------------
# TREND HIGHER HIGH LOWER LOW ANALYZER - GÜNCELLENMİŞ
# ---------------------------

class TrendHigherHighLowerLowAnalyzer:
    """
    TREND_HH_LL.py dosyasındaki mantığı kullanan güncellenmiş analyzer.
    Pivot High/Low tespiti ve ZigZag mantığı ile trend analizi yapar.
    """
    def __init__(self, lb=5, rb=5):
        self.lb = lb
        self.rb = rb
    
    def pivot_high(self, highs, i):
        if i - self.lb < 0 or i + self.rb >= len(highs):
            return np.nan
        window = highs[i - self.lb:i + self.rb + 1]
        if highs[i] == window.max():
            return highs[i]
        return np.nan
    
    def pivot_low(self, lows, i):
        if i - self.lb < 0 or i + self.rb >= len(lows):
            return np.nan
        window = lows[i - self.lb:i + self.rb + 1]
        if lows[i] == window.min():
            return lows[i]
        return np.nan
    
    def find_previous(self, i, hl_arr, zz_arr):
        current_hl = hl_arr[i]
        if np.isnan(current_hl):
            return np.nan, np.nan, np.nan, np.nan
        
        target1 = -1 if current_hl == 1 else 1
        loc1 = np.nan
        j = i - 1
        while j >= 0:
            if not np.isnan(hl_arr[j]) and hl_arr[j] == target1 and not np.isnan(zz_arr[j]):
                loc1 = zz_arr[j]
                break
            j -= 1
        
        j -= 1
        target2 = current_hl
        loc2 = np.nan
        while j >= 0:
            if not np.isnan(hl_arr[j]) and hl_arr[j] == target2 and not np.isnan(zz_arr[j]):
                loc2 = zz_arr[j]
                break
            j -= 1
        
        j -= 1
        target3 = -1 if current_hl == 1 else 1
        loc3 = np.nan
        while j >= 0:
            if not np.isnan(hl_arr[j]) and hl_arr[j] == target3 and not np.isnan(zz_arr[j]):
                loc3 = zz_arr[j]
                break
            j -= 1
        
        j -= 1
        target4 = current_hl
        loc4 = np.nan
        while j >= 0:
            if not np.isnan(hl_arr[j]) and hl_arr[j] == target4 and not np.isnan(zz_arr[j]):
                loc4 = zz_arr[j]
                break
            j -= 1
        
        return loc1, loc2, loc3, loc4
    
    def process_stock(self, df):
        n = len(df)
        df = df.copy().reset_index(drop=True)
        
        df['ph'] = np.nan
        df['pl'] = np.nan
        df['hl'] = np.nan
        df['zz'] = np.nan
        df['res'] = np.nan
        df['sup'] = np.nan
        df['trend'] = np.nan
        
        highs = df['HIGH_TL'].values
        lows = df['LOW_TL'].values
        closes = df['CLOSING_TL'].values
        
        for i in range(n):
            ph_val = self.pivot_high(highs, i)
            pl_val = self.pivot_low(lows, i)
            df.at[i, 'ph'] = ph_val
            df.at[i, 'pl'] = pl_val
            
            if not np.isnan(ph_val):
                df.at[i, 'hl'] = 1
                df.at[i, 'zz'] = ph_val
            elif not np.isnan(pl_val):
                df.at[i, 'hl'] = -1
                df.at[i, 'zz'] = pl_val
            else:
                df.at[i, 'hl'] = np.nan
                df.at[i, 'zz'] = np.nan
            
            if not np.isnan(df.at[i, 'hl']):
                j = i - 1
                while j >= 0 and np.isnan(df.at[j, 'hl']):
                    j -= 1
                if j >= 0:
                    prev_hl = df.at[j, 'hl']
                    prev_zz = df.at[j, 'zz']
                    if df.at[i, 'hl'] == -1 and prev_hl == -1 and (df.at[i, 'zz'] > prev_zz):
                        df.at[i, 'zz'] = np.nan
                    if df.at[i, 'hl'] == 1 and prev_hl == 1 and (df.at[i, 'zz'] < prev_zz):
                        df.at[i, 'zz'] = np.nan
                
                if j >= 0:
                    if df.at[i, 'hl'] == -1 and df.at[j, 'hl'] == 1 and (df.at[i, 'zz'] > df.at[j, 'zz']):
                        df.at[i, 'hl'] = np.nan
                        df.at[i, 'zz'] = np.nan
                    if df.at[i, 'hl'] == 1 and df.at[j, 'hl'] == -1 and (df.at[i, 'zz'] < df.at[j, 'zz']):
                        df.at[i, 'hl'] = np.nan
                        df.at[i, 'zz'] = np.nan
        
        for i in range(n):
            current_zz = df.at[i, 'zz']
            if not np.isnan(df.at[i, 'hl']):
                a = current_zz
                b, c, d, e = self.find_previous(i, df['hl'].values, df['zz'].values)
            else:
                a, b, c, d, e = np.nan, np.nan, np.nan, np.nan, np.nan
            
            _hh = False
            _ll = False
            _hl = False
            _lh = False
            
            if not np.isnan(a) and not np.isnan(b) and not np.isnan(c) and not np.isnan(d):
                _hh = (a > b and a > c and c > b and c > d)
                _ll = (a < b and a < c and c < b and c < d)
            
            if not np.isnan(a) and not np.isnan(b) and not np.isnan(c) and not np.isnan(d) and not np.isnan(e):
                _hl = ((a >= c and (b > c and b > d and d > c and d > e)) or 
                       (a < b and a > c and b < d))
                _lh = ((a <= c and (b < c and b < d and d < c and d < e)) or 
                       (a > b and a < c and b > d))
            
            if _lh:
                df.at[i, 'res'] = current_zz
            elif i > 0:
                df.at[i, 'res'] = df.at[i - 1, 'res']
            else:
                df.at[i, 'res'] = np.nan
            
            if _hl:
                df.at[i, 'sup'] = current_zz
            elif i > 0:
                df.at[i, 'sup'] = df.at[i - 1, 'sup']
            else:
                df.at[i, 'sup'] = np.nan
            
            if not np.isnan(df.at[i, 'res']) and closes[i] > df.at[i, 'res']:
                df.at[i, 'trend'] = 1
            elif not np.isnan(df.at[i, 'sup']) and closes[i] < df.at[i, 'sup']:
                df.at[i, 'trend'] = -1
            elif i > 0:
                df.at[i, 'trend'] = df.at[i - 1, 'trend']
            else:
                df.at[i, 'trend'] = np.nan
            
            if df.at[i, 'trend'] == 1 and _hh:
                df.at[i, 'res'] = current_zz
            if df.at[i, 'trend'] == -1 and _lh:
                df.at[i, 'res'] = current_zz
            if df.at[i, 'trend'] == 1 and _hl:
                df.at[i, 'sup'] = current_zz
            if df.at[i, 'trend'] == -1 and _ll:
                df.at[i, 'sup'] = current_zz
        
        return df
    
    def get_last_signal(self, df: pd.DataFrame):
        try:
            df['HIGH_TL'] = pd.to_numeric(df['HIGH_TL'], errors='coerce')
            df['LOW_TL'] = pd.to_numeric(df['LOW_TL'], errors='coerce')
            df['CLOSING_TL'] = pd.to_numeric(df['CLOSING_TL'], errors='coerce')
            
            processed_df = self.process_stock(df)
            
            signals = []
            for i in range(1, len(processed_df)):
                prev_trend = processed_df.at[i-1, 'trend']
                curr_trend = processed_df.at[i, 'trend']
                
                if not np.isnan(prev_trend) and not np.isnan(curr_trend):
                    if prev_trend == -1 and curr_trend == 1:
                        signals.append({'index': i, 'type': 'Buy'})
                    elif prev_trend == 1 and curr_trend == -1:
                        signals.append({'index': i, 'type': 'Sell'})
            
            if signals:
                last_signal = signals[-1]
                last_signal_type = last_signal['type']
                last_signal_candle_distance = len(processed_df) - 1 - last_signal['index']
            else:
                last_signal_type = None
                last_signal_candle_distance = None
            
            return last_signal_type, last_signal_candle_distance
            
        except Exception as e:
            print(f"Error in TrendHigherHighLowerLowAnalyzer: {e}")
            return None, None

# ---------------------------
# LINEAR REGRESSION SLOPE ANALYZER
# ---------------------------

class LinearRegressionSlopeAnalyzer:
    def __init__(self, curve_length: int = 68, slope_length: int = 99, signal_length: int = 10):
        self.curve_length = curve_length
        self.slope_length = slope_length
        self.signal_length = signal_length
    
    def calculate_linear_regression(self, prices: pd.Series, window: int) -> pd.Series:
        try:
            def linreg_value(y_values):
                if len(y_values) < window or y_values.isna().any():
                    return np.nan
                x = np.arange(len(y_values))
                coeffs = np.polyfit(x, y_values, 1)
                return coeffs[0] * (len(y_values) - 1) + coeffs[1]
            
            return prices.rolling(window=window).apply(linreg_value, raw=False)
        except Exception as e:
            print(f"Error in calculate_linear_regression: {e}")
            return pd.Series(np.full(len(prices), np.nan), index=prices.index)
    
    def calculate_ema(self, data: pd.Series, window: int) -> pd.Series:
        try:
            return data.ewm(span=window, adjust=False).mean()
        except Exception as e:
            print(f"Error in calculate_ema: {e}")
            return pd.Series(np.full(len(data), np.nan), index=data.index)
    
    def calculate_sma(self, data: pd.Series, window: int) -> pd.Series:
        try:
            return data.rolling(window=window).mean()
        except Exception as e:
            print(f"Error in calculate_sma: {e}")
            return pd.Series(np.full(len(data), np.nan), index=data.index)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            df['CLOSING_TL'] = pd.to_numeric(df['CLOSING_TL'], errors='coerce')
            
            df['lrc'] = self.calculate_linear_regression(df['CLOSING_TL'], self.curve_length)
            df['lrs'] = df['lrc'].diff()
            df['slrs'] = self.calculate_ema(df['lrs'], self.slope_length)
            df['alrs'] = self.calculate_sma(df['slrs'], self.signal_length)
            
            return df
        except Exception as e:
            print(f"Error in calculate_indicators: {e}")
            return pd.DataFrame({
                'lrc': np.full(len(df), np.nan),
                'lrs': np.full(len(df), np.nan),
                'slrs': np.full(len(df), np.nan),
                'alrs': np.full(len(df), np.nan)
            }, index=df.index)
    
    def get_last_signal(self, df: pd.DataFrame):
        try:
            df = self.calculate_indicators(df)
            
            condition1 = pd.Series(df['slrs'] > df['slrs'].shift(1), index=df.index).fillna(False)
            condition2 = pd.Series(df['slrs'].shift(1) <= df['alrs'].shift(1), index=df.index).fillna(False)
            condition3 = pd.Series(df['slrs'] > df['alrs'], index=df.index).fillna(False)
            
            buy_condition = condition1 & condition2 & condition3
            
            condition4 = pd.Series(df['slrs'] < df['slrs'].shift(1), index=df.index).fillna(False)
            condition5 = pd.Series(df['slrs'].shift(1) >= df['alrs'].shift(1), index=df.index).fillna(False)
            condition6 = pd.Series(df['slrs'] < df['alrs'], index=df.index).fillna(False)
            
            sell_condition = condition4 & condition5 & condition6
            
            buy_signals = df[buy_condition].index.tolist()
            sell_signals = df[sell_condition].index.tolist()
            
            all_signals = [(idx, 'BUY') for idx in buy_signals] + [(idx, 'SELL') for idx in sell_signals]
            all_signals.sort(key=lambda x: x[0])
            
            if all_signals:
                last_signal_idx, last_signal_type = all_signals[-1]
                last_signal_candle_distance = len(df) - last_signal_idx - 1
            else:
                last_signal_type = None
                last_signal_candle_distance = None
            
            return last_signal_type, last_signal_candle_distance
        except Exception as e:
            print(f"Error in LinearRegressionSlopeAnalyzer: {e}")
            return None, None

# ---------------------------
# RMT ANALYZER
# ---------------------------

class RMTAnalyzer:
    def __init__(self, Len2: int = 14, pmom: int = 65, nmom: int = 40):
        self.Len2 = Len2
        self.pmom = pmom
        self.nmom = nmom
    
    def rma(self, series: pd.Series, period: int) -> pd.Series:
        try:
            return series.ewm(alpha=1/period, adjust=False).mean()
        except Exception as e:
            print(f"Error in rma: {e}")
            return pd.Series(np.full(len(series), np.nan), index=series.index)
    
    def compute_rsi(self, close: pd.Series, period: int) -> pd.Series:
        try:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = self.rma(gain, period)
            avg_loss = self.rma(loss, period)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                rs = avg_gain / avg_loss
            rs = rs.replace([np.inf, -np.inf], np.nan)
            
            rsi = 100 - (100 / (1 + rs))
            rsi[avg_loss == 0] = 100
            rsi[avg_gain == 0] = 0
            
            return rsi
        except Exception as e:
            print(f"Error in compute_rsi: {e}")
            return pd.Series(np.full(len(close), np.nan), index=close.index)
    
    def get_last_signal(self, df: pd.DataFrame):
        try:
            df = df.copy()
            df['DATE'] = pd.to_datetime(df['DATE'])
            df = df.sort_values(by='DATE').reset_index(drop=True)
            df['CLOSING_TL'] = df['CLOSING_TL'].astype(str).str.replace(',', '.').astype(float)
            
            df['RSI'] = self.compute_rsi(df['CLOSING_TL'], self.Len2)
            df['EMA5'] = df['CLOSING_TL'].ewm(span=5, adjust=False).mean()
            df['EMA5_change'] = df['EMA5'].diff()
            
            df['p_mom'] = (df['RSI'].shift(1) < self.pmom) & (df['RSI'] > self.pmom) & (df['RSI'] > self.nmom) & (df['EMA5_change'] > 0)
            df['n_mom'] = (df['RSI'] < self.nmom) & (df['EMA5_change'] < 0)
            
            df['p_mom'] = df['p_mom'].fillna(False)
            df['n_mom'] = df['n_mom'].fillna(False)
            
            signals = []
            state = None
            
            for i, row in df.iterrows():
                if row['p_mom']:
                    if state != "GİRİŞ":
                        state = "GİRİŞ"
                        signals.append((row['DATE'], "GİRİŞ", row['CLOSING_TL']))
                elif row['n_mom']:
                    if state != "ÇIKIŞ":
                        state = "ÇIKIŞ"
                        signals.append((row['DATE'], "ÇIKIŞ", row['CLOSING_TL']))
            
            if signals:
                last_signal_date, last_signal_type, _ = signals[-1]
                matching_rows = df[df['DATE'] == last_signal_date]
                if not matching_rows.empty:
                    pos = matching_rows.index[0]
                    consecutive_bars = len(df) - pos - 1
                else:
                    consecutive_bars = None
            else:
                last_signal_type = None
                consecutive_bars = None
            
            if last_signal_type == "GİRİŞ":
                last_signal_type = "Buy"
            elif last_signal_type == "ÇIKIŞ":
                last_signal_type = "Sell"
            
            return last_signal_type, consecutive_bars
            
        except Exception as e:
            print(f"Error in RMTAnalyzer: {e}")
            return None, None

# ---------------------------
# KALMAN FILTER ANALYZER
# ---------------------------

class KalmanTrendAnalyzer:
    def __init__(self, short_len=50, long_len=150, retest_sig=False):
        self.short_len = short_len
        self.long_len = long_len
        self.retest_sig = retest_sig
        
    def kalman_filter(self, src: np.array, length: int, R: float = 0.01, Q: float = 0.1) -> np.array:
        try:
            n = len(src)
            estimate = np.zeros(n)
            error_est = np.ones(n)
            error_meas = R * length
            kalman_gain = np.zeros(n)
            prediction = np.zeros(n)
            
            if n > 0:
                estimate[0] = src[0] if not np.isnan(src[0]) else 0
                error_est[0] = 1.0
            
            for i in range(1, n):
                if np.isnan(src[i]):
                    estimate[i] = estimate[i-1]
                    continue
                    
                prediction[i] = estimate[i-1]
                kalman_gain[i] = error_est[i-1] / (error_est[i-1] + error_meas)
                estimate[i] = prediction[i] + kalman_gain[i] * (src[i] - prediction[i])
                error_est[i] = (1 - kalman_gain[i]) * error_est[i-1] + Q / length
                
            return estimate
        except Exception as e:
            print(f"Error in kalman_filter: {e}")
            return np.zeros(len(src))
    
    def calculate_atr(self, high: np.array, low: np.array, close: np.array, period: int = 200) -> np.array:
        try:
            high_low = high - low
            high_close = np.abs(high - np.roll(close, 1))
            low_close = np.abs(low - np.roll(close, 1))
            
            tr = np.maximum(high_low, np.maximum(high_close, low_close))
            tr[0] = high_low[0]
            
            atr = pd.Series(tr).rolling(window=period, min_periods=1).mean().values
            return atr * 0.5
        except Exception as e:
            print(f"Error in calculate_atr: {e}")
            return np.zeros(len(high))
    
    def _detect_retest_signals(self, high: np.array, low: np.array, close: np.array, 
                              trend_changes: list, atr: np.array) -> list:
        try:
            retest_signals = []
            
            for i, change_idx in enumerate(trend_changes):
                if change_idx >= len(high) - 1:
                    continue
                    
                if i < len(trend_changes) - 1:
                    end_idx = trend_changes[i + 1]
                else:
                    end_idx = len(high)
                
                for j in range(change_idx + 1, min(end_idx, len(high))):
                    if j > 0:
                        if (high[j] < (low[change_idx] + atr[change_idx]) and 
                            high[j-1] >= (low[change_idx] + atr[change_idx])):
                            retest_signals.append({
                                'index': j-1,
                                'type': 'RETEST_SELL'
                            })
                        
                        if (low[j] > (high[change_idx] - atr[change_idx]) and 
                            low[j-1] <= (high[change_idx] - atr[change_idx])):
                            retest_signals.append({
                                'index': j-1,
                                'type': 'RETEST_BUY'
                            })
            
            return retest_signals
        except Exception as e:
            print(f"Error in _detect_retest_signals: {e}")
            return []
    
    def get_last_signal(self, df: pd.DataFrame):
        try:
            if len(df) < max(self.short_len, self.long_len):
                return None, None
            
            df['CLOSING_TL'] = pd.to_numeric(df['CLOSING_TL'], errors='coerce')
            df['HIGH_TL'] = pd.to_numeric(df['HIGH_TL'], errors='coerce')
            df['LOW_TL'] = pd.to_numeric(df['LOW_TL'], errors='coerce')
            
            close = df['CLOSING_TL'].values
            high = df['HIGH_TL'].values
            low = df['LOW_TL'].values
            
            short_kalman = self.kalman_filter(close, self.short_len)
            long_kalman = self.kalman_filter(close, self.long_len)
            
            trend_up = np.zeros(len(close), dtype=bool)
            for i in range(len(close)):
                if not np.isnan(short_kalman[i]) and not np.isnan(long_kalman[i]):
                    trend_up[i] = short_kalman[i] > long_kalman[i]
                else:
                    trend_up[i] = False
            
            trend_changes = []
            for i in range(1, len(trend_up)):
                if trend_up[i] != trend_up[i-1]:
                    trend_changes.append(i)
            
            signals = []
            for idx in trend_changes:
                if trend_up[idx]:
                    signals.append({'index': idx, 'type': 'BUY'})
                else:
                    signals.append({'index': idx, 'type': 'SELL'})
            
            if self.retest_sig:
                atr = self.calculate_atr(high, low, close)
                retest_signals = self._detect_retest_signals(high, low, close, trend_changes, atr)
                signals.extend(retest_signals)
            
            if signals:
                last_signal = signals[-1]
                last_signal_type = last_signal['type']
                last_signal_candle_distance = len(df) - 1 - last_signal['index']
            else:
                last_signal_type = None
                last_signal_candle_distance = None
            
            return last_signal_type, last_signal_candle_distance
        except Exception as e:
            print(f"Error in KalmanTrendAnalyzer: {e}")
            return None, None

# ---------------------------
# TREF ANALYZER
# ---------------------------

class TREFAnalyzer:
    def __init__(self, rsi_period: int = 13, pmom: int = 68, nmom: int = 40):
        self.rsi_period = rsi_period
        self.pmom = pmom
        self.nmom = nmom
    
    def rma(self, series: pd.Series, period: int) -> pd.Series:
        try:
            return series.ewm(alpha=1/period, adjust=False).mean()
        except Exception as e:
            print(f"Error in rma: {e}")
            return pd.Series(np.full(len(series), np.nan), index=series.index)
    
    def compute_rsi(self, close: pd.Series, period: int) -> pd.Series:
        try:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = self.rma(gain, period)
            avg_loss = self.rma(loss, period)
            
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            rsi[avg_loss == 0] = 100
            rsi[avg_gain == 0] = 0
            return rsi
        except Exception as e:
            print(f"Error in compute_rsi: {e}")
            return pd.Series(np.full(len(close), np.nan), index=close.index)
    
    def compute_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
        try:
            TP = (high + low + close) / 3
            money_flow = TP * volume
            diff = TP.diff()
            
            pos_flow = money_flow.where(diff > 0, 0)
            neg_flow = money_flow.where(diff < 0, 0)
            
            sum_pos = pos_flow.rolling(window=period).sum()
            sum_neg = neg_flow.rolling(window=period).sum()
            
            money_flow_ratio = sum_pos / sum_neg.replace(0, np.nan)
            mfi = 100 - (100 / (1 + money_flow_ratio))
            mfi = mfi.fillna(100)
            return mfi
        except Exception as e:
            print(f"Error in compute_mfi: {e}")
            return pd.Series(np.full(len(close), np.nan), index=close.index)
    
    def get_last_signal(self, df: pd.DataFrame):
        try:
            df = df.copy()
            
            df['CLOSING_TL'] = pd.to_numeric(df['CLOSING_TL'].astype(str).str.replace(',', '.'), errors='coerce')
            df['HIGH_TL'] = pd.to_numeric(df['HIGH_TL'].astype(str).str.replace(',', '.'), errors='coerce')
            df['LOW_TL'] = pd.to_numeric(df['LOW_TL'].astype(str).str.replace(',', '.'), errors='coerce')
            df['VOLUME_TL'] = pd.to_numeric(df['VOLUME_TL'].astype(str).str.replace(',', '.'), errors='coerce')
            
            df.rename(columns={
                'DATE': 'Date',
                'CLOSING_TL': 'Close',
                'LOW_TL': 'Low',
                'HIGH_TL': 'High',
                'VOLUME_TL': 'Volume'
            }, inplace=True)
            
            df['RSI'] = self.compute_rsi(df['Close'], self.rsi_period)
            df['MFI'] = self.compute_mfi(df['High'], df['Low'], df['Close'], df['Volume'], self.rsi_period)
            df['rsi_mfi'] = (df['RSI'] + df['MFI']) / 2
            
            df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
            df['EMA5_change'] = df['EMA5'].diff()
            
            condition1 = pd.Series(df['rsi_mfi'].shift(1) < self.pmom, index=df.index).fillna(False)
            condition2 = pd.Series(df['rsi_mfi'] > self.pmom, index=df.index).fillna(False)
            condition3 = pd.Series(df['rsi_mfi'] > self.nmom, index=df.index).fillna(False)
            condition4 = pd.Series(df['EMA5_change'] > 0, index=df.index).fillna(False)
            
            df['p_mom'] = condition1 & condition2 & condition3 & condition4
            
            condition5 = pd.Series(df['rsi_mfi'] < self.nmom, index=df.index).fillna(False)
            condition6 = pd.Series(df['EMA5_change'] < 0, index=df.index).fillna(False)
            
            df['n_mom'] = condition5 & condition6
            
            signals = []
            for _, row in df.iterrows():
                if row['p_mom'] and (not signals or signals[-1][1] != "AL"):
                    signals.append((row['Date'], "AL", row['Close']))
                elif row['n_mom'] and (not signals or signals[-1][1] != "SAT"):
                    signals.append((row['Date'], "SAT", row['Close']))
            
            if signals:
                last_signal_date, last_signal_type, _ = signals[-1]
                last_date = df['Date'].iloc[-1]
                last_signal_candle_distance = (last_date - last_signal_date).days
            else:
                last_signal_type = None
                last_signal_candle_distance = None
            
            if last_signal_type == "AL":
                last_signal_type = "Buy"
            elif last_signal_type == "SAT":
                last_signal_type = "Sell"
            
            return last_signal_type, last_signal_candle_distance
        except Exception as e:
            print(f"Error in TREFAnalyzer: {e}")
            return None, None

# ---------------------------
# MAIN FUNCTION
# ---------------------------

def main():
    try:
        df = pd.read_excel('hisse_verileri_2y.xlsx')
        print(f"Data loaded: {len(df)} rows, {df['CODE'].nunique()} different stocks")
        
        for col in ['CLOSING_TL', 'HIGH_TL', 'LOW_TL', 'VOLUME_TL']:
            if col in df.columns:
                non_numeric = df[~df[col].apply(lambda x: str(x).replace('.', '').replace(',', '').isdigit())]
                if not non_numeric.empty:
                    print(f"Warning: Found {len(non_numeric)} non-numeric values in column {col}")
                    print(f"Sample values: {non_numeric[col].head(5).tolist()}")
    except FileNotFoundError:
        print("ERROR: 'hisse_verileri_2y.xlsx' file not found!")
        return
    
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values(['CODE', 'DATE']).reset_index(drop=True)
    
    vidya_strategy = VIDYAStrategy(len_V=2, len_VHist=5, strat_loop=1, end_loop=60, long_t=40, short_t=10)
    trend_analyzer = TrendHigherHighLowerLowAnalyzer()
    linrs_analyzer = LinearRegressionSlopeAnalyzer()
    rmt_analyzer = RMTAnalyzer()
    kalman_analyzer = KalmanTrendAnalyzer(retest_sig=False)
    tref_analyzer = TREFAnalyzer()
    
    results = []
    unique_tickers = df['CODE'].unique()
    print(f"Total {len(unique_tickers)} stocks to analyze...")
    
    for i, ticker in enumerate(unique_tickers):
        print(f"Analyzing: {ticker} ({i+1}/{len(unique_tickers)})")
        
        stock_data = df[df['CODE'] == ticker].copy()
        stock_data = stock_data.sort_values('DATE')
        stock_data = stock_data.reset_index(drop=True)
        
        if len(stock_data) < 100:
            print(f"  - {ticker}: Insufficient data ({len(stock_data)} days)")
            continue
        
        try:
            vidya_signal, vidya_bars_ago = vidya_strategy.get_last_signal(stock_data['CLOSING_TL'])
            trend_signal, trend_bars_ago = trend_analyzer.get_last_signal(stock_data)
            linrs_signal, linrs_bars_ago = linrs_analyzer.get_last_signal(stock_data)
            rmt_signal, rmt_bars_ago = rmt_analyzer.get_last_signal(stock_data)
            kalman_signal, kalman_bars_ago = kalman_analyzer.get_last_signal(stock_data)
            tref_signal, tref_bars_ago = tref_analyzer.get_last_signal(stock_data)
            
            wma_result = analyze_stock_wma(stock_data, ticker, n=146)
            lrb_result = analyze_stock_linreg(stock_data, ticker, period=105)
            finh_result = analyze_stock_finh(stock_data, ticker, period=110, yukdus=True)
            
            wma_pct_diff = wma_result['WMA_Percentage Diff'] if wma_result else np.nan
            wma_duration = wma_result['WMA_Filter Duration'] if wma_result else np.nan
            lrb_pct_diff = lrb_result['LRB_Percentage Diff'] if lrb_result else np.nan
            lrb_duration = lrb_result['LRB_Filter Duration'] if lrb_result else np.nan
            finh_pct_diff = finh_result['FINH_Percentage Diff'] if finh_result else np.nan
            finh_duration = finh_result['FINH_Filter Duration'] if finh_result else np.nan
            
            results.append({
                "Stock Symbol": ticker,
                "VIDYA Signal": vidya_signal,
                "VIDYA Signal – Bars Ago": vidya_bars_ago,
                "Trend Signal": trend_signal,
                "Trend Signal – Bars Ago": trend_bars_ago,
                "linRS Signal": linrs_signal,
                "linRS Signal – Bars Ago": linrs_bars_ago,
                "RMT Signal": rmt_signal,
                "RMT Signal – Bars Ago": rmt_bars_ago,
                "Kalman Filter Signal": kalman_signal,
                "Kalman Signal – Bars Ago": kalman_bars_ago,
                "TREF Signal": tref_signal,
                "TREF Signal – Bars Ago": tref_bars_ago,
                "WMA Percentage Diff": wma_pct_diff,
                "WMA Filter Duration": wma_duration,
                "LRB Percentage Diff": lrb_pct_diff,
                "LRB Filter Duration": lrb_duration,
                "FINH Percentage Diff": finh_pct_diff,
                "FINH Filter Duration": finh_duration
            })
            
            print(f"  - {ticker}: Completed")
            
        except Exception as e:
            print(f"  - {ticker}: Error - {str(e)}")
            continue
    
    results_df = pd.DataFrame(results)
    output_filename = 'unified_analysis.xlsx'
    
    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        results_df.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        buy_format = workbook.add_format({
            'bg_color': '#C6EFCE',
            'font_color': '#006100',
            'border': 1
        })
        
        sell_format = workbook.add_format({
            'bg_color': '#FFC7CE',
            'font_color': '#9C0006',
            'border': 1
        })
        
        pct_format = workbook.add_format({
            'num_format': '0.00',
            'border': 1
        })
        
        int_format = workbook.add_format({
            'num_format': '0',
            'border': 1
        })
        
        for idx, col in enumerate(results_df.columns):
            worksheet.set_column(idx, idx, 15)
        
        for col_num, value in enumerate(results_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        signal_columns = [1, 3, 5, 7, 9, 11]
        
        for col_idx in signal_columns:
            worksheet.conditional_format(1, col_idx, len(results_df), col_idx, {
                'type': 'text',
                'criteria': 'containing',
                'value': 'Buy',
                'format': buy_format
            })
            
            worksheet.conditional_format(1, col_idx, len(results_df), col_idx, {
                'type': 'text',
                'criteria': 'containing',
                'value': 'Sell',
                'format': sell_format
            })
        
        pct_columns = [13, 15, 17]
        for col_idx in pct_columns:
            for row_idx in range(1, len(results_df) + 1):
                if not pd.isna(results_df.iloc[row_idx-1, col_idx]):
                    worksheet.write(row_idx, col_idx, results_df.iloc[row_idx-1, col_idx], pct_format)
            
            worksheet.conditional_format(1, col_idx, len(results_df), col_idx, {
                'type': 'cell',
                'criteria': '>',
                'value': 0,
                'format': buy_format
            })
            
            worksheet.conditional_format(1, col_idx, len(results_df), col_idx, {
                'type': 'cell',
                'criteria': '<',
                'value': 0,
                'format': sell_format
            })
        
        duration_columns = [14, 16, 18]
        for col_idx in duration_columns:
            for row_idx in range(1, len(results_df) + 1):
                if not pd.isna(results_df.iloc[row_idx-1, col_idx]):
                    worksheet.write(row_idx, col_idx, results_df.iloc[row_idx-1, col_idx], int_format)
        
        # Freeze panes at A2 to keep header visible
        worksheet.freeze_panes(1, 0)
        
        # Add autofilter to header row
        worksheet.autofilter(0, 0, len(results_df), len(results_df.columns) - 1)
    
    print(f"\nAnalysis completed!")
    print(f"Results saved to '{output_filename}' file.")
    print(f"Total {len(results)} stocks analyzed.")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for signal_col in ["VIDYA Signal", "Trend Signal", "linRS Signal", "RMT Signal", "Kalman Filter Signal", "TREF Signal"]:
        if signal_col in results_df.columns:
            buy_count = (results_df[signal_col] == "Buy").sum()
            sell_count = (results_df[signal_col] == "Sell").sum()
            total = buy_count + sell_count
            if total > 0:
                buy_pct = (buy_count / total) * 100
                sell_pct = (sell_count / total) * 100
                print(f"\n{signal_col}:")
                print(f"  Buy Signals:  {buy_count:3d} ({buy_pct:5.1f}%)")
                print(f"  Sell Signals: {sell_count:3d} ({sell_pct:5.1f}%)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
