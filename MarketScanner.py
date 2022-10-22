from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.contract import Contract
import datetime as dt
import time
import numpy as np
import pandas as pd
import os

class TestApp(EClient, EWrapper):

    def __init__(self):
        EClient.__init__(self, self)

    def get_data(self):
        super().reqScannerParameters()

    def scannerParameters(self, xml:str):
        super().scannerParameters(xml)
        print('heyhehysfhdfs')


class MyClient(EClient):
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)
        self.reqScannerParameters()



class MyWrapper(EWrapper):

    def scannerParameters(self, xml:str):
        super().scannerParameters(xml)
        print('heyhehysfhdfs')


    def nextValidId(self, orderId):
        print(f'The next valid order ID is {orderId}')
        self.init_ID = orderId



class MyApp(MyWrapper, MyClient):
    def __init__(self):
        MyWrapper.__init__(self)
        MyClient.__init__(self, wrapper = self)





app = MyApp()
app.connect('127.0.0.1', 7497, 1117)


app.run()