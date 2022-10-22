import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import os
from getting_intraday_data import get_data
import math
from stock_class import Stock
import time
def load_data(stock):

    get_data(stock)
    dates =  os.listdir(f'{stock}_intraday_data')

    data_list = []

    for day in dates:
        d = pd.read_csv(f'{stock}_intraday_data' + '/' + f'{day}')

        if len(d):
            data_list.append(d)


    return data_list




def find_peaks(stock):

    data = load_data(stock)

    high_list = []
    low_list = []
    high_times = []
    low_times = []

    high_rel = []
    low_rel = []

    for day in data:

        day = day.iloc[:, :17]
        day = day.dropna()
        day.reset_index(drop = True)


        opens = day['open'].tolist()
        highs = day['high'].tolist()
        lows = day['low'].tolist()
        closes = day['close'].tolist()
        times = day['label'].tolist()


        highs = [c for c in highs if math.isnan(c) == False]

        peak = max(highs)
        valley = min(lows)

        high_ind = highs.index(peak)
        low_ind = lows.index(valley)
        peak_time = times[high_ind]
        valley_time = times[low_ind]

        peak_time = peak_time[:-3]

        valley_time = valley
        high_list.append(peak)
        low_list.append(valley)
        high_times.append(peak_time)
        low_times.append(valley_time)

        #find relative percantage moves

        high_rel.append((peak - opens[0]) / opens[0] * 100)
        low_rel.append((valley - opens[0]) / opens[0] * 100)





    return high_list, low_list, high_times, low_times, high_rel, low_rel


def RSI_analysis(ticker, list_of_params):

    rsi_buy, rsi_sell, rsi_short, rsi_cover, shortsma, longsma = list_of_params
    buys = []
    sells = []
    pnl = 0

    long = False
    short = False
    count = 0
    data_list = load_data(ticker)
    for day in data_list:
        count += 1
        today_stock = Stock('SPY', day)
        today_RSI = today_stock.RSI()
        print(today_RSI)
        today_high = today_stock.high
        today_low = today_stock.low
        shortSMA = today_stock.SMA(shortsma)
        longSMA = today_stock.SMA(longsma)
        for i, item in enumerate(today_RSI):

            if not long and not short:

                if item <= rsi_buy and shortSMA[i] > longSMA[i]:

                    midpoint = (today_high[i] + today_low[i]) / 2
                    buys.append(midpoint)
                    pnl -= midpoint
                    print(f'bought at {midpoint}')

                    long = True

                elif item >= rsi_short and shortSMA[i] < longSMA[i]:

                    midpoint = (today_high[i] + today_low[i]) / 2
                    sells.append(midpoint)
                    pnl += midpoint
                    print(f'sold at {midpoint}')

                    short = True

            elif (long and item > rsi_sell and shortSMA[i] < longSMA[i]):
                midpoint = (today_high[i] + today_low[i]) / 2
                sells.append(midpoint)
                pnl += midpoint
                print(f'sold at {midpoint}')

                long = False


            elif (short and item < rsi_cover and shortSMA[i] > longSMA[i]):
                midpoint = (today_high[i] + today_low[i]) / 2
                buys.append(midpoint)
                pnl -= midpoint
                print(f'bought at {midpoint}')

                short = False

        #
        # if long:
        #     midpoint = (today_high[i] + today_low[i]) / 2
        #     sells.append(midpoint)
        #     pnl += midpoint
        #     print(f'sold at {midpoint}')
        #
        #     long = False
        #
        # elif short:
        #     midpoint = (today_high[i] + today_low[i]) / 2
        #     buys.append(midpoint)
        #     pnl -= midpoint
        #     print(f'bought at {midpoint}')
        #
        #     short = False
        #print(f'Day {count} completed with PnL of {pnl}')
        print(f'COMPLETED SIM\n'
              f'{rsi_buy}\n'
              f'{rsi_sell}\n'
              f'{rsi_short}\n'
              f'{rsi_cover}\n'
              f'{shortsma}\n'
              f'{longsma}')

    return rsi_buy, rsi_sell, rsi_short, rsi_cover, shortsma, longsma, buys, sells, pnl


buy_range = np.arange(15,36, 2).tolist()
sell_range = np.arange(50, 75, 2).tolist()
short_range = np.arange(65, 86, 2).tolist()
cover_range = np.arange(25, 50, 2).tolist()

shortsmarange = [9, 15, 20, 50]
longsmarange = [50, 75, 100, 150, 200]

my_input = [buy_range, sell_range, short_range, cover_range, shortsmarange, longsmarange]
def create_matrix(input_params):
    buy_range, sell_range, short_range, cover_range, shortsmarange, longsmarange = input_params

    option_list = []
    counter = 0

    option = [c[0] for c in input_params]

    opt = []
    while len(buy_range):
        a = buy_range[0]
        option[0] = a
        option_list.append(option)
        del buy_range[0]
        print(option_list)

        while len(sell_range):
            b = sell_range[0]
            option[1] = b
            option_list.append(option)
            del sell_range[0]

            while len(short_range):
                c = short_range[0]
                option[2] = c
                option_list.append(option)
                del short_range[0]

                while len(cover_range):
                    d = cover_range[0]
                    option[3] = d
                    option_list.append(option)
                    del cover_range[0]

                    while len(shortsmarange):
                        e = shortsmarange[0]
                        option[4] = e
                        option_list.append(option)
                        del shortsmarange[0]


                        while len(longsmarange):
                            f = longsmarange[0]
                            option[5] = f
                            option_list.append(option)
                            del longsmarange[0]

    return option_list


listboi = create_matrix(my_input)

print('final list', listboi)
