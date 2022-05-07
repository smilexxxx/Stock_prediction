#import necessary dependencies
import numpy as np
import pandas as pd
import tensorflow as ts
import streamlit as st
from matplotlib import pyplot as plt

from pandas_datareader import data, wb
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

#define start of data bracket and end
start = '2010-01-01'
end = '2021-12-31'
st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start, end)

#describing the data
st.subheader('Data from 2010 - 2021')
st.write(df.describe())

#data visualization
st.subheader('Closing Price vs Time')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time with 100 moving average')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time with 100 & 200 moving average')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

#split data to training set(70%) and prediction test(30%)
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

#scale down the data
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

#split data to x_train and y_train
#x_train = []
#y_train = []
#for i in range(100, data_training_array.shape[0]):
#    x_train.append(data_training_array[i - 100: i])
#    y_train.append(data_training_array[i, 0])

#convert x_train and y_train arrays into numpy arrays
#x_train, y_train = np.array(x_train), np.array(y_train)

#load model
model = load_model('keras_model.h5')

#prediction
#first get previsous 100 days
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)

#scale down the data
input_data=scaler.fit_transform(final_df)

#define x_test and y_test
x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

#convert to numpy arrays
x_test, y_test = np.array(x_test), np.array(y_test)

#making predictions
y_predicted = model.predict(x_test)

#scaling
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#plotting final graph
st.subheader('Predicted price vs Original price')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'g', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)