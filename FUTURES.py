import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from iexfinance.stocks import get_historical_intraday
import datetime as dt
import os
os.environ['IEX_TOKEN'] = 'sk_e7c00c9a73144c10aeee3815d2b8b7f0'

# data = get_historical_intraday('NQH21', dt.datetime.today() - dt.timedelta(1))
# print(data)

