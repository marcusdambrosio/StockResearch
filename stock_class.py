# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 14:19:45 2020

@author: Marcus
"""

#imports 

import yfinance as yf
from yahoofinancials import YahooFinancials
from datetime import datetime, date, timedelta
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

from binance.client import Client
from BinanceKeys import BinanceKey1
import pandas as pd



''' Building a class that will hopefully work as a web-based stock graphing tool 3/1/2020'''

class Crypto:
    
    def __init__(self, symbol, candle_period, start, end):
        
        self.period = candle_period
        self.sym = symbol
        
        api_key = BinanceKey1['api_key']
        api_secret = BinanceKey1['api_secret']

        client = Client(api_key, api_secret)
        
        
        candle_num = [c for c in candle_period if c.isnumeric()]
        candle_str = [c for c in candle_period if c.isalpha()]
        
        time_per_dict = {'m' : 'm',
                         'M' : 'm',
                         'min' : 'm',
                         'Min' : 'm',
                         'minute' : 'm',
                         'Minute' : 'm',
                         'MINUTE' : 'm',
                         
                         'h' : 'h',
                         'H' : 'h',
                         'hr' : 'h',
                         'Hr' : 'h',
                         'HR' : 'h', 
                         'hour' : 'h',
                         'Hour' : 'h',
                         'HOUR' : 'h',
                         
                         'd' : 'd',
                         'D' : 'd',
                         'day' : 'd', 
                         'Day' : 'd',
                         'DAY' : 'd',
                         
                         'w' : 'w',
                         'W' : 'w',
                         'wk' : 'w',
                         'Wk' : 'w',
                         'WK' : 'w',
                         'week' : 'w',
                         'Week' : 'w',
                         'WEEK' : 'w'
                         }
       
        try:
            candle_str = time_per_dict[candle_str[0]]
          
        except:
            print('invalid interval')
          
        constructed_interval = candle_num[0] + candle_str
            
        self.data = pd.DataFrame(client.get_historical_klines(symbol = symbol, interval = constructed_interval, start_str = start, end_str = end))
        self.data.columns = ['open time', 'open', 'high', 'low', 'close',  
                             'volume', 'close time' , 'quote asset volume', 'trades', 
                             'taker buy base asset volume', 'taker by quote asset volume', 'ignore']
       
        print('data constructed')


        self.open = self.data['open'].astype(float)
        self.high = self.data['high'].astype(float)
        self.low = self.data['low'].astype(float)
        self.close = self.data['close'].astype(float)
        self.volume = self.data['volume'].astype(float) 
          
 
    
    
    def SMA(self, period):
        closes = self.close
        sma = closes.rolling(window = period, min_periods = 1).mean()
        return sma   

              
    def EMA(self, period):
        closes = self.close
        ema = closes.ewm(span = period, min_periods = 1).mean()
        return ema

       
    def MACD(self):
        closes = self.close
        macd = closes.ewm(span = 12, min_periods = 1).mean() - closes.ewm(span = 26, min_periods = 1).mean()
        return macd
    
    
    def stochastic(self):
        closes = self.close
        lows = self.low
        highs = self.high     
        h14 = np.array(highs.rolling(window = 14, min_periods = 1).max())
        l14 = np.array(lows.rolling(window = 14, min_periods = 1).min())
        print(type(closes[0]))
        stoch = (np.array(closes) - l14) / (h14 - l14) * 100       
        return stoch
    
    def RSI(self):
        closes = self.close
        rsi = []
        gains = []
        losses = []
        avg_loss = 1
        avg_gain = 1
        for i, item in enumerate(closes):
       
            if i == 0:
                i = 1
            
            change = item - closes[i-1]
            if i <= 13:                
                if change >= 0:
                    gains.append(change)
                    
                elif change < 0:
                    losses.append(change)
                    
                
                    
                rsi.append(50)
                    
            
                    
            else:
            
                if change >= 0:
                    gains.append(np.abs(change))
                    del gains[0]
                    avg_gain = np.mean(gains)
                    #avg_gain = np.sum(gains) / 14
                    
                elif change < 0:
                    losses.append(np.abs(change))
                    del losses[0]
                    avg_loss = np.mean(losses)
                    #avg_loss = np.sum(losses) / 14
                
                try:    
                    RS = avg_gain / avg_loss
                except:
                    next
                    
                if not len(gains) + len(losses) == 14:
                    print('length error')
    
                rsi.append(100 - 100 / (1 + RS))
                
        return rsi
    


#btc = Crypto('BTCUSDT' , '15m' , '2020-02-25', '2020-03-05')





def find_p_change(crypto_object):
    
    closes = crypto_object.close
    diffs = []    
    
    i = 0
    while i < len(closes) - 2:
        diffs.append((closes[i+1] - closes[i]) / closes[i] * 100)
        i += 1
    return diffs    
        
def find_log_change(crypto_object):
    
    closes = crypto_object.close
    log_diffs = []
    
    i = 0
    while i < len(closes) - 2:
        log_diffs.append((np.log(closes[i+1]) - np.log(closes[i])) / np.log(closes[i]))
        i += 1
    return log_diffs 

def visualize_changes(crypto_object):
    
    diffs = find_p_change(crypto_object)
    log_diffs = find_log_change(crypto_object)
    
    plt.hist(diffs , bins = 100)
 
    
    

def filter_diffs(threshold):
    
    diffs = find_p_change(btc)
    
    fdiffs = [c for c in diffs if np.abs(c) > threshold]
    
    index = []
    for i in fdiffs:
        index.append(diffs.index(i))
        
        
    return diffs, index



def df_constructor(crypto_object):
    filtered_diffs, index = filter_diffs(.075)
    
    o, h, l, c, v = [], [], [], [], []

    for i in index:
        
        o.append(crypto_object.open[i])
        h.append(crypto_object.high[i])
        l.append(crypto_object.low[i])
        c.append(crypto_object.close[i])
        v.append(crypto_object.volume[i])
        
    df = pd.DataFrame() 
    df['open'] = o
    df['high'] = h
    df['low'] = l
    df['close'] = c
    df['volume'] = v
    
    return df, index


def pull_indicators(crypto_object):
    
    df, index = df_constructor(btc)
    
    SMA7 = crypto_object.SMA(7)
    SMA20 = crypto_object.SMA(20)
    SMA50 = crypto_object.SMA(50)
    SMA100 = crypto_object.SMA(100)
    SMA200 = crypto_object.SMA(200)
    
    EMA7 = crypto_object.EMA(7)
    EMA20 = crypto_object.EMA(20)
    EMA50 = crypto_object.EMA(50)
    EMA100 = crypto_object.EMA(100)
    EMA200 = crypto_object.EMA(200)
    
    
    closes = df['close']
    num_above = []
    print(len(closes), len(index))
    for i, item in enumerate(index):
        counter = 0
    
        for p in [SMA7, SMA20, SMA50, SMA100, SMA200, EMA7, EMA20, EMA50, EMA100, EMA200]:
            
            if closes[i] > p[item]:
                counter += 1
                
            elif closes[i] < p[item]:
                counter -= 1
                
            else:
                print('something wrong')
                
        num_above.append(counter)

    return num_above, index



def find_alpha(crypto_object):
    ''' makes bar graph of average pnl vs number of SMA / EMA the price is above '''
    closes = crypto_object.close
    
    num_above, index = pull_indicators(crypto_object)
    
    pnl = []
    
    for i in index:
        
        pnl.append(closes[i+1] - closes[i])
        
    
    minten = []
    mineight = []
    minsix = []
    minfour = []
    mintwo = []
    zero = []
    two = []
    four = []
    six = []
    eight = []
    ten = []

    for i , item in enumerate(num_above):
        
        if item == -10:
            minten.append(pnl[i])              
        elif item == -8:
            mineight.append(pnl[i])
        elif item == -6:
            minsix.append(pnl[i])
        elif item == -4:
            minfour.append(pnl[i])
        elif item == -2:
            mintwo.append(pnl[i])
        elif item == 0:
           zero.append(pnl[i])
        elif item == 2:
           two.append(pnl[i])
        elif item == 4:
           four.append(pnl[i])
        elif item == 6:
           six.append(pnl[i])
        elif item == 8:
           eight.append(pnl[i])
        elif item == 10:
           ten.append(pnl[i])
          
        else:
            print('broken shit')
           
    avg_index = np.arange(-10 , 11, step = 2)
    print(avg_index)
    avg_values = []
   
    for i in [minten, mineight, minsix, minfour, mintwo, zero, two, four, six, eight, ten]:
        avg_values.append(np.mean(i))
        
       
    plt.bar(avg_index,avg_values)
   


def analyze(crypto_object):
    full_data = crypto_object.data
    
    diffs, index = filter_diffs(.075)
    new_df = pd.DataFrame()
    for i, item in enumerate(index):
        
        new_df = new_df.append(full_data.iloc[item, :])
        
        if i > 1000:
            break
        
    return new_df
        


    
class Stock:
    
    def __init__(self, ticker, input_data):
        
        financials = YahooFinancials(ticker)
    
        summary = financials.get_summary_data(ticker)[f'{ticker}']
            #summary arrives in a single key dictionary so must pull data out

        self.data = input_data
        self.open = input_data['open'].astype(float)
        self.high = input_data['high'].astype(float)
        self.low = input_data['low'].astype(float)
        self.close = input_data['close'].astype(float)
        self.volume = input_data['volume'].astype(float)


        #all attributes are defined in the order that they appear in the summary
        '''
        self.prev_close = summary['previousClose']
        self.open = summary['regularMarketOpen']
        self.twtyfour_hour_vol = summary['volume24Hr'] #Volume over 24 hours including pre and after market
        self.daily_high = summary['regularMarketDayHigh'] #High during market hours
        self.avg_vol = summary['averageDailyVolume10Day'] #Average volume over last 10  days
        self.daily_low = summary['regularMarketDayLow'] #Low during market hours
        self.daily_vol = summary['regularMarketVolume'] #Volume during market hours
        self.mkt_cap = summary['marketCap']
        self.yearly_high = summary['fiftyTwoWeekHigh']
        self.yearly_low = summary['fiftyTwoWeekLow']
        
        self.shares_outstanding = financials.get_num_shares_outstanding()
        
        #self made attributes
        
        self.relative_vol = self.daily_vol / self.avg_vol
        #figure out how to make a float
        

        #get data
        end = str(date.today())
        start = str(date.today() - timedelta(days = 50)) #Configure how many days back data goes, start at 50
        
        self.data_daily = pd.DataFrame(financials.get_historical_price_data(start, end, 'daily')[f'{ticker}']['prices'])
        self.data_weekly = pd.DataFrame(financials.get_historical_price_data(start, end, 'weekly')[f'{ticker}']['prices'])
        self.data_monthly =  pd.DataFrame(financials.get_historical_price_data(start, end, 'monthly')[f'{ticker}']['prices'])
        #self.data_yearly = pd.DataFrame(financials.get_historical_price_data(start, end, 'yearly')[f'{ticker}']['prices'])
            #All data will come as DataFrame with columns date , high, low, open, close, volume, adjclose, formatted_date
        '''
        
    ''' All method calculations '''
    
    #first, define x axis
    
    
    def ichimoku(self, short_period = 9, mid_period = 26, long_period = 52, offset = 26, plot = False):
        #ichimoku will require special plotting function because it is not a simple line
        dates = data['formatted_date']
        
        ichimoku = pd.DataFrame()
        

        period1_high = self.high.rolling(short_period).max()
        period1_low = self.low.rolling(short_period).max()
        ichimoku['tenkan_sen'] = (period1_high + period1_low) / 2
        
        period2_high = self.high.rolling(mid_period).max()
        period2_low = self.low.rolling(mid_period).max()
        ichimoku['kijun_sen'] = (period2_high + period2_low) / 2
        
        ichimoku['senkou_span_a'] = ((ichimoku['tenkan_sen_{}'.format(short_period)] + ichimoku['kijun_sen_{}'.format(mid_period)]) /2 ).shift(offset)
    
        period3_high = self.high.rolling(long_period).max()
        period3_low = self.high.rolling(long_period).max()
        ichimoku['senkou_span_b' ]= ((period3_high + period3_low)/2).shift(offset)
        
        
        ichimoku['chikou_span'] = self.close.shift(-offset)
        
        if plot == 'True':
            #Somehow this will triger only when the indicator is selected
            plt.plot(dates, ichimoku['senkou_span_a'], color = 'g')
            plt.plot(dates, ichimoku['senkou_span_b'], color = 'r')
            plt.fill_between(dates, ichimoku['senkou_span_a'], ichimoku['senkou_span_b'], where = ichimoku['senkou_span_a'] >= ichimoku['senkou_span_b'], facecolor = 'green', alpha = .5, interpolate = True)
            plt.fill_between(dates, ichimoku['senkou_span_a'], ichimoku['senkou_span_b'], where = ichimoku['senkou_span_a'] <= ichimoku['senkou_span_b'], facecolor = 'red', alpha = .5, interpolate = True)
            
            plt.plot(dates, ichimoku['tenkan_sen'], color = 'b', label = 'Tenkan-sen')
            plt.plot(dates, ichimoku['kijun_sen'], color = 'r', linewidth = 1.5, label = 'Kijun-sen')
            plt.plot(dates, ichimoku['chikou_span'], color = 'g', linewidth = 1.5, label = 'Chikou Span')
            plt.legend()
            
        return ichimoku

    def SMA(self, period):
        closes = self.close
        sma = closes.rolling(window = period, min_periods = 1).mean()
        return sma   

              
    def EMA(self, period):
        closes = self.close
        ema = closes.ewm(span = period, min_periods = 1).mean()
        return ema

       
    def MACD(self):
        closes = self.close
        macd = closes.ewm(span = 12, min_periods = 1).mean() - closes.ewm(span = 26, min_periods = 1).mean()
        return macd
    
    
    def stochastic(self):
        closes = self.close
        lows = self.low
        highs = self.high     
        h14 = np.array(highs.rolling(window = 14, min_periods = 1).max())
        l14 = np.array(lows.rolling(window = 14, min_periods = 1).min())

        stoch = (np.array(closes) - l14) / (h14 - l14) * 100       
        return stoch
    
    
    def RSI(self):
        closes = self.close
        rsi = []
        gains = []
        losses = []
        avg_loss = 1
        avg_gain = 1
        for i, item in enumerate(closes):
       
            if i == 0:
                i = 1
            
            change = item - closes[i-1]
            if i <= 13:                
                if change >= 0:
                    gains.append(change)
                    
                elif change < 0:
                    losses.append(change)
                    
                
                    
                rsi.append(50)
                    
            
                    
            else:
            
                if change >= 0:
                    gains.append(np.abs(change))
                    del gains[0]
                    avg_gain = np.mean(gains)
                    #avg_gain = np.sum(gains) / 14
                    
                elif change < 0:
                    losses.append(np.abs(change))
                    del losses[0]
                    avg_loss = np.mean(losses)
                    #avg_loss = np.sum(losses) / 14
                
                try:    
                    RS = avg_gain / avg_loss
                except:
                    next
                    
                if not len(gains) + len(losses) == 14:
                    print('length error')
    
                rsi.append(100 - 100 / (1 + RS))
                
        return rsi
      
        
        
    
