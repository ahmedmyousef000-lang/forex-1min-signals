from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

class TechnicalIndicators:
    @staticmethod
    def calculate_ema(data, period):
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_sma(data, period):
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_rsi(data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - signal_line
        return macd, signal_line, hist
    
    @staticmethod
    def calculate_stochastic(high, low, close, k_period=14, d_period=3):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return k, d
    
    @staticmethod
    def calculate_bollinger_bands(data, period=20, std=2):
        middle = data.rolling(window=period).mean()
        std_dev = data.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower
    
    @staticmethod
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def calculate_adx(high, low, close, period=14):
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr = TechnicalIndicators.calculate_atr(high, low, close, period)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx
    
    @staticmethod
    def calculate_cci(high, low, close, period=14):
        tp = (high + low + close) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean())
        return (tp - sma) / (0.015 * mad)

class YFinanceDataProvider:
    def __init__(self):
        self.symbol_mapping = {
            'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X', 'USDJPY': 'USDJPY=X',
            'USDCHF': 'USDCHF=X', 'AUDUSD': 'AUDUSD=X', 'USDCAD': 'USDCAD=X',
            'NZDUSD': 'NZDUSD=X', 'EURGBP': 'EURGBP=X'
        }
    
    def get_data(self, symbol, timeframe='1min', days=5):
        if symbol not in self.symbol_mapping:
            return None
        yf_symbol = self.symbol_mapping[symbol]
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(interval='1m', start=start_date, end=end_date)
            if df.empty:
                return None
            df = df.reset_index()
            df.columns = df.columns.str.lower()
            if 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'timestamp'})
            elif 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'})
            if 'volume' not in df.columns:
                df['volume'] = 0
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None

class ForexSignalGenerator:
    def __init__(self):
        self.major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD', 'EURGBP']
        self.timeframe = '1min'
        self.min_confidence = 60
        
    def calculate_all_indicators(self, df):
        try:
            df['EMA_9'] = TechnicalIndicators.calculate_ema(df['close'], 9)
            df['EMA_21'] = TechnicalIndicators.calculate_ema(df['close'], 21)
            df['EMA_50'] = TechnicalIndicators.calculate_ema(df['close'], 50)
            df['SMA_200'] = TechnicalIndicators.calculate_sma(df['close'], 200)
            df['RSI'] = TechnicalIndicators.calculate_rsi(df['close'], 14)
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = TechnicalIndicators.calculate_macd(df['close'])
            df['Slowk'], df['Slowd'] = TechnicalIndicators.calculate_stochastic(df['high'], df['low'], df['close'])
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = TechnicalIndicators.calculate_bollinger_bands(df['close'])
            df['ATR'] = TechnicalIndicators.calculate_atr(df['high'], df['low'], df['close'])
            df['ADX'] = TechnicalIndicators.calculate_adx(df['high'], df['low'], df['close'])
            df['CCI'] = TechnicalIndicators.calculate_cci(df['high'], df['low'], df['close'])
            return df
        except Exception as e:
            return df
    
    def strategy_trend_following(self, df):
        signal, strength = 0, 0
        try:
            current, previous = df.iloc[-1], df.iloc[-2]
            if pd.isna(current['ADX']) or pd.isna(current['EMA_9']):
                return signal, strength
            if current['ADX'] > 20:
                if (previous['EMA_9'] <= previous['EMA_21'] and current['EMA_9'] > current['EMA_21'] and current['close'] > current['EMA_50']):
                    signal, strength = 1, min(current['ADX'] / 40 * 100, 100)
                elif (previous['EMA_9'] >= previous['EMA_21'] and current['EMA_9'] < current['EMA_21'] and current['close'] < current['EMA_50']):
                    signal, strength = -1, min(current['ADX'] / 40 * 100, 100)
        except:
            pass
        return signal, strength
    
    def strategy_momentum(self, df):
        signal, strength = 0, 0
        try:
            current = df.iloc[-1]
            if pd.isna(current['RSI']) or pd.isna(current['Slowk']):
                return signal, strength
            if current['RSI'] < 30 and current['Slowk'] < 20 and current['Slowk'] > current['Slowd']:
                signal, strength = 1, (30 - current['RSI']) * 2
            elif current['RSI'] > 70 and current['Slowk'] > 80 and current['Slowk'] < current['Slowd']:
                signal, strength = -1, (current['RSI'] - 70) * 2
        except:
            pass
        return signal, strength
    
    def strategy_macd_divergence(self, df):
        signal, strength = 0, 0
        try:
            current, previous = df.iloc[-1], df.iloc[-2]
            if pd.isna(current['MACD']) or pd.isna(current['MACD_signal']):
                return signal, strength
            if (previous['MACD'] <= previous['MACD_signal'] and current['MACD'] > current['MACD_signal'] and current['MACD'] < 0):
                signal, strength = 1, min(abs(current['MACD_hist']) * 10000, 100)
            elif (previous['MACD'] >= previous['MACD_signal'] and current['MACD'] < current['MACD_signal'] and current['MACD'] > 0):
                signal, strength = -1, min(abs(current['MACD_hist']) * 10000, 100)
        except:
            pass
        return signal, strength
    
    def strategy_bollinger_bounce(self, df):
        signal, strength = 0, 0
        try:
            current, previous = df.iloc[-1], df.iloc[-2]
            if pd.isna(current['BB_lower']) or pd.isna(current['BB_upper']):
                return signal, strength
            if (previous['close'] <= previous['BB_lower'] and current['close'] > current['BB_lower'] and current['RSI'] < 40):
                bb_range = current['BB_middle'] - current['BB_lower']
                if bb_range > 0:
                    signal, strength = 1, ((current['BB_middle'] - current['close']) / bb_range * 100)
            elif (previous['close'] >= previous['BB_upper'] and current['close'] < current['BB_upper'] and current['RSI'] > 60):
                bb_range = current['BB_upper'] - current['BB_middle']
                if bb_range > 0:
                    signal, strength = -1, ((current['close'] - current['BB_middle']) / bb_range * 100)
        except:
            pass
        return signal, strength
    
    def strategy_cci_extremes(self, df):
        signal, strength = 0, 0
        try:
            current, previous = df.iloc[-1], df.iloc[-2]
            if pd.isna(current['CCI']):
                return signal, strength
            if previous['CCI'] < -100 and current['CCI'] > -100:
                signal, strength = 1, min(abs(current['CCI']), 100)
            elif previous['CCI'] > 100 and current['CCI'] < 100:
                signal, strength = -1, min(abs(current['CCI']), 100)
        except:
            pass
        return signal, strength
    
    def strategy_multi_timeframe(self, df):
        signal, strength = 0, 0
        try:
            current = df.iloc[-1]
            if any(pd.isna(current[col]) for col in ['EMA_9', 'EMA_21', 'EMA_50', 'SMA_200']):
                return signal, strength
            if (current['EMA_9'] > current['EMA_21'] > current['EMA_50'] and current['close'] > current['SMA_200']):
                signal, strength = 1, 80
            elif (current['EMA_9'] < current['EMA_21'] < current['EMA_50'] and current['close'] < current['SMA_200']):
                signal, strength = -1, 80
        except:
            pass
        return signal, strength
    
    def calculate_stop_loss_take_profit(self, df, signal, atr_multiplier=1.0, risk_reward=2):
        try:
            current = df.iloc[-1]
            atr, price = current['ATR'], current['close']
            if pd.isna(atr):
                return None, None
            if signal == 1:
                return price - (atr * atr_multiplier), price + (atr * atr_multiplier * risk_reward)
            elif signal == -1:
                return price + (atr * atr_multiplier), price - (atr * atr_multiplier * risk_reward)
            return None, None
        except:
            return None, None
    
    def calculate_risk_reward_ratio(self, entry, stop_loss, take_profit):
        try:
            if stop_loss and take_profit:
                risk = abs(entry - stop_loss)
                reward = abs(take_profit - entry)
                if risk > 0:
                    return round(reward / risk, 2)
            return 0
        except:
            return 0
    
    def generate_signal(self, df):
        df = self.calculate_all_indicators(df)
        if len(df) < 200:
            return {'signal': 'NO DATA', 'confidence': 0, 'entry_price': 0, 'stop_loss': None, 'take_profit': None, 'active_strategies': [], 'indicators': {}, 'risk_reward': 0, 'strategy_count': 0}
        
        strategies = {
            'Trend Following': self.strategy_trend_following(df),
            'Momentum': self.strategy_momentum(df),
            'MACD': self.strategy_macd_divergence(df),
            'Bollinger Bands': self.strategy_bollinger_bounce(df),
            'CCI Extremes': self.strategy_cci_extremes(df),
            'Multi-Timeframe': self.strategy_multi_timeframe(df)
        }
        
        total_signal, total_strength, active_strategies, strategy_count = 0, 0, [], 0
        for name, (sig, strength) in strategies.items():
            if sig != 0:
                total_signal += sig
                total_strength += strength
                strategy_count += 1
                direction = "BUY" if sig == 1 else "SELL"
                active_strategies.append(f"{name} → {direction} ({round(strength, 1)}%)")
        
        if total_signal >= 3:
            final_signal, confidence = "BUY", min(total_strength / max(strategy_count, 1), 100) if strategy_count > 0 else 0
        elif total_signal <= -3:
            final_signal, confidence = "SELL", min(total_strength / max(strategy_count, 1), 100) if strategy_count > 0 else 0
        else:
            final_signal, confidence = "NEUTRAL", 0
        
        signal_value = 1 if final_signal == "BUY" else (-1 if final_signal == "SELL" else 0)
        stop_loss, take_profit = self.calculate_stop_loss_take_profit(df, signal_value)
        current = df.iloc[-1]
        risk_reward = self.calculate_risk_reward_ratio(current['close'], stop_loss, take_profit)
        
        return {
            'signal': final_signal,
            'confidence': round(confidence, 2),
            'entry_price': round(current['close'], 5),
            'stop_loss': round(stop_loss, 5) if stop_loss else None,
            'take_profit': round(take_profit, 5) if take_profit else None,
            'active_strategies': active_strategies,
            'indicators': {
                'RSI': round(current['RSI'], 2) if not pd.isna(current['RSI']) else 0,
                'MACD': round(current['MACD'], 6) if not pd.isna(current['MACD']) else 0,
                'ADX': round(current['ADX'], 2) if not pd.isna(current['ADX']) else 0,
                'Stochastic': round(current['Slowk'], 2) if not pd.isna(current['Slowk']) else 0,
                'CCI': round(current['CCI'], 2) if not pd.isna(current['CCI']) else 0,
                'ATR': round(current['ATR'], 5) if not pd.isna(current['ATR']) else 0
            },
            'risk_reward': risk_reward,
            'strategy_count': strategy_count,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

app = Flask(__name__)
signal_generator = ForexSignalGenerator()
data_provider = YFinanceDataProvider()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/signals')
def get_signals():
    signals = []
    for pair in signal_generator.major_pairs:
        df = data_provider.get_data(pair, '1min', days=5)
        if df is not None and len(df) >= 200:
            signal = signal_generator.generate_signal(df)
            signal['pair'] = pair
            signals.append(signal)
    return jsonify(signals)

@app.route('/api/signal/<pair>')
def get_single_signal(pair):
    df = data_provider.get_data(pair, '1min', days=5)
    if df is not None and len(df) >= 200:
        signal = signal_generator.generate_signal(df)
        signal['pair'] = pair
        return jsonify(signal)
    return jsonify({'error': 'No data available'})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
