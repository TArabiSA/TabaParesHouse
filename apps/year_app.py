import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from streamlit_lottie import st_lottie
import math
from data.year_data import create_yeartable


sf= create_yeartable


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
        df = sf
        st.write(df)
    with right_column:
        st_lottie(lottie_coding, height = 300, key ="coding")
# First Graph
    st.title("### Plot Data :chart_with_upwards_trend:")
    st.markdown("### ")
    st.title("Sale's on 2007 to 2021 chart")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    df = sf
    plt.figure(figsize=(12,6))
    plt.title('Sale Price History')
    plt.plot(df.Sale)
    plt.plot(df.Sale, 'b')
    plt.xlabel('YEAR', fontsize = 12)
    plt.ylabel('Sale Price PESO(₱)', fontsize = 12)
    st.pyplot()
# Second Graph

    st.title("Sale's vs Time chart with 3 years moving average")
    df = sf
    ma3 = df.Sale.rolling(3).mean()
    fig = plt.plot(figsize=(12,6))
    plt.plot(ma3)
    plt.plot(ma3, 'r')
    plt.plot(df.Sale)
    plt.plot(df.Sale, 'b')
    plt.title('Sale Price History')
    plt.xlabel('YEAR', fontsize = 12)
    plt.ylabel('Sale Price PESO(₱)', fontsize = 12)
    plt.legend(['Sale','MA3'], loc = 'upper left')
    st.pyplot(fig)

# Third Graph

    st.title("Sale's vs Time chart with 3 years and 6 years moving average")
    df = sf
    ma3 = df.Sale.rolling(3).mean()
    ma6 = df.Sale.rolling(6).mean()
    fig = plt.plot(figsize=(12,6))
    plt.plot(ma3)
    plt.plot(ma3, 'r')
    plt.plot(ma6)
    plt.plot(ma6, 'orange')
    plt.plot(df.Sale)
    plt.plot(df.Sale,'b')
    plt.title('Sale Price History')
    plt.xlabel('YEAR', fontsize = 12)
    plt.ylabel('Sale Price PESO(₱)', fontsize = 12)
    plt.legend(['Sale','MA3','MA6'], loc = 'upper left')
    st.pyplot(fig)




    
# Fourth Graph
# Load my model
    st.title('Prediction Sale vs Original Sale  ')
    data = df.filter(['Sale'])
    dataset = data.values
    training_data_len = math.ceil( len(dataset) * .7)
    training_data_len
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    scaled_data

#spliting data into  x_train and y_train
    train_data = scaled_data[0:training_data_len , :]
    x_train = []
    y_train = []

    for i in range(3, len(train_data)):
        x_train.append(train_data[i-3:i, 0])
        y_train.append(train_data[i, 0])
    if i<= 3:
        print(x_train)
        print(y_train)
        print()

    x_train, y_train = np.array(x_train), np.array(y_train)
    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], 3, 1))
    x_train.shape

# load my model
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

    model.fit(x_train, y_train, batch_size=1, epochs=10)
    
    

# test part

    test_data = scaled_data[training_data_len -3: , :]
    test_data.shape
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(3, len(test_data)):
        x_test.append(test_data[i -3:i, 0])
       
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    rmse = np.sqrt( np.mean( predictions - y_test )**2 )
    rmse


      

    train = data[:training_data_len]
    prediction = data[training_data_len:]
    prediction['Predictions'] = predictions
    fig = plt.figure(figsize=(12,6))
    plt.title('Sale Price History')
    plt.plot(df)
    plt.xlabel('Date', fontsize = 12)
    plt.ylabel('Sale Price PESO(₱)', fontsize = 12)
    plt.plot(train['Sale'])
    plt.plot(prediction[[ 'Predictions']])
    plt.legend(['TEST','TRAIN','PREDICTION'], loc = 'upper left')
    st.pyplot(fig)
    
    year = sf
    newdf = year.filter(['Sale'])
    last_3_years = newdf[-3:].values
    last_3_years_scaled = scaler.transform(last_3_years)
    X_test = [] 
    X_test.append(last_3_years)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    pred_sale = model.predict(X_test)
    pred_sale = scaler.inverse_transform(pred_sale)
    print(pred_sale)

# prediction
    st.subheader("Prediction for 2022 Sale's")
    st.write(pred_sale)
