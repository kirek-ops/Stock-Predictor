import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf

from datetime import datetime
from pandas_datareader import data as pdr 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

yf.pdr_override()

# Load Data
company = 'AAPL'
start = datetime(2012, 1, 1)
end = datetime(2020, 1, 1)

data = pdr.get_data_yahoo(company, start=start, end=end)
