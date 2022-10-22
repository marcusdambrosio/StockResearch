import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import pandas as pd
import datetime as dt
from iexfinance.stocks import get_historical_data
from iexfinance.stocks import Stock
import os

os.environ['output_format'] ='pandas'
os.environ['IEX_TOKEN'] = 'sk_9bdbb9855bb441da890e392fbe146d16'

start = dt.datetime(2019, 1,  1)
end = dt.datetime.today()
'''
data = get_historical_data('SPY', start, end, output_format = 'pandas')
data.to_csv('spy_data.csv')
'''
data = pd.read_csv('spy_data.csv')

open = data['open']
high = data['high']
low = data['low']
close = data['close']
volume = data['volume']
date = data.index


def graph_overnight_diffs(plot = True):
    diffs = []

    for i, item in enumerate(open):

        try:
            diff = (item - close[i-1]) / close[i-1] * 100
            diffs.append(diff)
        except:
            diffs.append(0)

    if plot:

        plt.plot(date, diffs, linewidth = .5)
        plt.xticks(rotation = 'vertical')
        plt.ylabel('Overnight Change (%)')
        plt.show()

    return diffs #percent

def graph_daily_diffs(plot = True):
    diffs = []

    for i, item in enumerate(open):

        try:
            diff = (close[i] - item) / item * 100
            diffs.append(diff)
        except:
            diffs.append(0)

    if plot:

        plt.plot(date, diffs, linewidth=.5)
        plt.xticks(rotation = 'vertical')
        plt.ylabel('Daily Change (%)')
        plt.show()

    return diffs #percent

def night_day_corr():

    night_diff = graph_overnight_diffs(plot=False)
    day_diff = graph_daily_diffs(plot=False)

    plt.scatter(night_diff, day_diff, linewidth = .2)
    plt.plot([np.min(night_diff), np.max(night_diff)], [np.min(day_diff), np.max(day_diff)], color = 'green', linewidth = .5)
    plt.plot([np.min(night_diff), np.max(night_diff)], [0,0] , color = 'blue', linewidth = .5)
    plt.plot([0,0] , [np.min(day_diff), np.max(day_diff)], color='blue', linewidth=.5)
    plt.ylabel('Day Change')
    plt.xlabel('Overnight Gap')

    plt.show()


def night_day_stats():

    night_diff = graph_overnight_diffs(plot=False)
    day_diff = graph_daily_diffs(plot=False)

    #NIGHT STATS
    pct_night_gap_up = [c for c in night_diff if c > 0 ]
    pct_night_gap_down = [c for c in night_diff if c <= 0]

    avg_night_up = np.mean(pct_night_gap_up)
    avg_night_down = np.mean(pct_night_gap_down)

    avg_night = np.mean(night_diff)
    median_night = np.median(night_diff)

    #DAY STATS
    pct_day_gap_up = [c for c in night_diff if c > 0]
    pct_day_gap_down = [c for c in night_diff if c <= 0]

    avg_day_up = np.mean(pct_day_gap_up)
    avg_day_down = np.mean(pct_day_gap_down)

    avg_day = np.mean(day_diff)
    median_day = np.median(day_diff)

    #DAY AFTER STATS
    pct_day_after_up = []
    pct_day_after_down = []
    for i, item in enumerate(night_diff):

        if item > 0:
            pct_day_after_up.append(day_diff[i])

        else:
            pct_day_after_down.append(day_diff[i])

    avg_day_after_up = np.mean(pct_day_after_up) #average day after gap up
    avg_day_after_down = np.mean(pct_day_after_down) #average day after gap down

    #finding chance of green/red day depending on night

    green_up = []
    red_up = []

    for i, item in enumerate(pct_day_after_up):

        if item > 0:
            green_up.append(item)

        else:
            red_up.append(item)


    green_down = []
    red_down = []

    for i, item in enumerate(pct_day_after_down):

        if item > 0:
            green_down.append(item)

        else:
            red_down.append(item)


    chance_green_up = len(green_up) / (len(green_up) + len(red_up)) * 100
    avg_green_up = np.mean(green_up)
    chance_red_up = 100 - chance_green_up
    avg_red_up = np.mean(red_up)

    chance_green_down = len(green_down) / (len(green_down) + len(red_down)) * 100
    avg_green_down = np.mean(green_down)
    chance_red_down = 100 - chance_green_down
    avg_red_down = np.mean(red_down)

    #CHOOSE RESULTS BELOW













