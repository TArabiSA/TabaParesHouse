from html.entities import html5
import streamlit as st
import numpy as np
import pandas as pd
import requests
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt
from data.week_data import create_weektable




def app():
 st.title('Data :page_facing_up:')   

 st.write("The following is the DataFrame of the `Taba Pares House` dataset.")

 def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#-------LOAD ASSETS------
 lottie_coding = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_o6spyjnc.json") 


 with st.container():
    
    left_column, right_column = st.columns(2)
    with left_column:
        df = create_weektable
        st.write(df)
    with right_column:
        st_lottie(lottie_coding, height = 300, key ="coding")
# First graph
    st.title("Plot Data :chart_with_upwards_trend:")
    st.markdown("###   ")
    st.subheader("Weekly Sale's from January 10 to March 27 2022 chart")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    df = pd.read_csv('C:/Users/yakul/mutipage/data/week.csv')
   
    plt.figure(figsize=(12,6))
    plt.title('Sale Price History')
    plt.plot(df['Sale'])
    plt.xlabel('WEEKS', fontsize = 12)
    plt.ylabel('Sale Price PESO(₱)', fontsize = 12)
    st.pyplot()

# Second Graph

    st.subheader("Sale's vs Time chart with 4 weeks moving average")
    df = pd.read_csv('C:/Users/yakul/mutipage/data/week.csv')
    ma4 = df.Sale.rolling(4).mean()
    ma4
    plt.figure(figsize = (12,6))
    plt.plot(df.Sale)
    plt.plot(ma4)
    plt.title('Sale Price History')
    plt.xlabel('WEEKS', fontsize = 12)
    plt.ylabel('Sale Price PESO(₱)', fontsize = 12)
    plt.legend(['Sale','MA4'], loc = 'upper left')
    st.pyplot()


 # Third Graph 
    
    st.subheader("Sale's vs Time chart with 4 weeks and 8 weeks moving average")
    df = pd.read_csv('C:/Users/yakul/mutipage/data/week.csv')
    ma4 = df.Sale.rolling(4).mean()
    ma4
    ma8 = df.Sale.rolling(8).mean()
    ma8
    plt.figure(figsize = (12,6))
    plt.plot(df.Sale)
    plt.plot(ma4)
    plt.plot(ma8)
    plt.title('Sale Price History')
    plt.xlabel('WEEKS', fontsize = 12)
    plt.ylabel('Sale Price PESO(₱)', fontsize = 12)
    plt.legend(['Sale','MA4', 'MA8'], loc = 'upper left')
    st.pyplot()


# Fourth Graph and load model
    st.subheader('Prediction Sale vs Original Sale  ')
    import math
    from tensorflow import keras
    from keras.models import load_model
    
    
    data = df.filter(['Sale'])
    dataset = data.values
    training_data_len = math.ceil( len(dataset) * .7)
    training_data_len

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    scaled_data

    train_data = scaled_data[0:training_data_len , :]
    x_train = []
    y_train = []

    for i in range(4, len(train_data)):
     x_train.append(train_data[i-4:i, 0])
     y_train.append(train_data[i, 0])
    if i<= 4:
     print(x_train)
     print(y_train)
     print()

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss= 'mean_squared_error')

    model.summary()
    
    
    model.fit(x_train, y_train, batch_size=1, epochs=100)


    test_data = scaled_data[training_data_len -4: , :]
    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(4, len(test_data)):
        x_test.append(test_data[i -4:i,0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    rmse = np.sqrt( np.mean( predictions - y_test )**2 )
    rmse

    train = data[:training_data_len]
    prediction = data[training_data_len:]
    prediction['Predictions'] = predictions
    plt.figure(figsize=(12,6))
    plt.title('Sale Price History')
    plt.plot(data)
    plt.xlabel('WEEKS', fontsize = 12)
    plt.ylabel('Sale Price PESO(₱)', fontsize = 12)
    plt.plot(train['Sale'])
    plt.plot(prediction[[ 'Predictions']])
    plt.legend(['TEST','TRAIN','PREDICTION'], loc = 'upper left')
    st.pyplot()

# Prediction april 3 22022

    weeks = pd.read_csv('C:/Users/yakul/mutipage/data/week.csv')
    newdf = weeks.filter(['Sale'])
    last_4_weeks = newdf[-4:].values
    last_4_weeks_scaled = scaler.transform(last_4_weeks)
    X_test = [] 
    X_test.append(last_4_weeks)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    pred_sale = model.predict(X_test)
    pred_sale = scaler.inverse_transform(pred_sale)
    print(pred_sale)
    
    
    st.subheader("Prediction for April 3 2022  Sale's")
    st.write(pred_sale)


 
        
  

