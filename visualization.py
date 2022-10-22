import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplfinance
import os

# def night_space(model_df, night_length = 60):
#     column_count = len(model_df.columns.tolist())
#     night = {}
#
#     # for column in model_df.columns.tolist():
#     #     night[f'{column}'] = np.zeros(night_length)
#
#     night = {'Open': np.zeros(night_length),
#                     'High': np.zeros(night_length),
#                     'Low': np.zeros(night_length),
#                     'Close': np.zeros(night_length),
#                     'Volume': np.zeros(night_length)}
#
#     return night

def night_space(night_length = 60):

    night = {'Open': np.zeros(night_length),
            'High': np.zeros(night_length),
            'Low': np.zeros(night_length),
            'Close': np.zeros(night_length),
            'Volume': np.zeros(night_length)}

    return night


def prepare_data(symbol, trades):

    datelist = os.listdir(f'{symbol}_intraday_data')
    dates = []

    cols = pd.read_csv('A_intraday_data/2019-05-22.csv').columns

    master_df = pd.DataFrame({'Open' : 0,
                             'High' : 0,
                             'Low' : 0,
                             'Close' : 0,
                             'Volume' : 0}, index = datelist)

    for day in datelist:
        current_data = pd.read_csv(f'{symbol}_intraday_data/{day}')

        if not len(current_data):

            master_df = master_df.append(night_space(), ignore_index = True)
            master_df = master_df.append(night_space(), ignore_index = True)

            temp_night = night_space()

            dates += np.zeros(2 * len(temp_night['Open'])).tolist()

            continue

        temp_night = night_space()
        dates += np.zeros(len(current_data['open'].tolist())).tolist()
        dates += np.zeros(len(temp_night['Open'])).tolist()

        current_data = {'Open': current_data['marketOpen'],
             'High': current_data['marketHigh'],
             'Low': current_data['marketLow'],
             'Close': current_data['marketClose'],
             'Volume': current_data['marketVolume']}

        # current_data = current_data.append(night_space(current_data))

        master_df = master_df.append(current_data, ignore_index=True)
        master_df = master_df.append(night_space(), ignore_index=True)
        print(master_df.tail())


    #
    # final_data = pd.DataFrame({'Open' : master_df['marketOpen'],
    #                'High' : master_df['marketHigh'],
    #                'Low' : master_df['marketLow'],
    #                'Close' : master_df['marketClose'],
    #                'Volume' : master_df['marketVolume']})


    # master_df.set_index(dates, inplace = True)
    master_df.index = dates
    # final_data.set_index(dates, inplace = True)

    return master_df, dates, trades

data, dates, trades = prepare_data('AAP', 0 )
print(data.head())
print(dates)

def visualize(symbol, trade_list):

    ohlc, dates, trades = prepare_data(trade_list)

    my_plot = plt.figure()
    ax1 = my_plot.add_sublot([1,1,1])

    new_candle_reference = np.arange(len(ohlc))
    old_ohlc = [col[0] for col in ohlc]
    total_ohlc = old_ohlc

    for index, time in enumerate(ohlc):

        current_ohlc = [col[index] for col in ohlc]

        if index < 250:

            if new_candle_reference%5 == 0:

                for i, item in enumerate(current_ohlc):
                    total_ohlc[i] = total_ohlc[i].append(item)

            else:

                if current_ohlc[1] > old_ohlc[1]:
                    old_ohlc[1] = current_ohlc[1]

                if current_ohlc[2] > old_ohlc[2]:
                    old_ohlc[2] = current_ohlc[2]

                if current_ohlc[3] < old_ohlc[3]:
                    olc_ohlc[3] = current_ohlc[3]

                if current_ohlc[4] < old_ohlc[4]:
                    old_ohlc[4] = current_ohlc[4]

                if current_ohlc[5] > old_ohlc[5]:
                    old_ohlc[5] = current_ohlc[5]

                total_ohlc[-1] = old_ohlc

        else:

            if new_candle_reference % 5 == 0:

                for i, item in enumerate(current_ohlc):
                    current_col = total_ohlc[i]
                    del current_col[0]
                    current_col.append(item)

                    total_ohlc[i] = current_col

            else:

                if current_ohlc[1] > old_ohlc[1]:
                    old_ohlc[1] = current_ohlc[1]

                if current_ohlc[2] > old_ohlc[2]:
                    old_ohlc[2] = current_ohlc[2]

                if current_ohlc[3] < old_ohlc[3]:
                    olc_ohlc[3] = current_ohlc[3]

                if current_ohlc[4] < old_ohlc[4]:
                    old_ohlc[4] = current_ohlc[4]

                if current_ohlc[5] > old_ohlc[5]:
                    old_ohlc[5] = current_ohlc[5]

                total_ohlc[-1] = old_ohlc

        #trade info
        # current_trade = trades[0]
        # trade_date = current_trade[0]
        # trade_time = current_trade[1]
        #
        # if current_date == trade_date and current_time == trade_time:
        #     plt.scatter(trade_time, trade_price)
        #
        ax1.clear()

        candlestick_ohlc(ax1, current_ohlc, colorup='#77d879', colordown='#db3f3f')
        ax1.grid()
        ax1.title(symbol)
        ax1.xlabel('Time')
        ax1.ylabel('Price')

        ax1.show()



