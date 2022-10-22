import pandas as pd
import numpy as np
import datetime as dt
import time

data = pd.read_csv('../GOOGL_intraday_data/2019-08-02.csv')

print(data.columns)

print(data['high'], data['marketHigh'])