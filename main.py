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

# Load data
company = 'AAPL'
start = datetime(2012, 1, 1)
end = datetime(2020, 1, 1)

data = pdr.get_data_yahoo(company, start=start, end=end)

# Prepare data 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build model
model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))

#Prediction of the next closest value 
model.add(Dense(units = 1)) 

model.compile(optimizer = 'adam', loss = 'mean_squeared_error')
model.fit(x_train, y_train, epochs = 25, batch_size = 32)


''' Test the model accurancy on existing data '''

# Load data
test_start = datetime(2020, 1, 1)
test_end = datetime(2022, 1, 1)

test_data = pdr.get_data_yahoo(company, test_start, test_end)
actual_values = test_data['Close'].values
 
total_dataset = pd.concat((data['Close'], test_data['Close']), axis = 0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Make some predictions on test data
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

