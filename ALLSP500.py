import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from corr_finder import save_sp500_tickers
from getting_intraday_data import get_data


ticker_list = save_sp500_tickers()
print(ticker_list)

for ticker in ticker_list:
    get_data(ticker)

    print(f'{ticker} data loaded')

