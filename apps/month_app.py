import streamlit as st
import numpy as np
import pandas as pd
import requests
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt
from data.month_data import create_monthtable

sf = create_monthtable

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
        df = create_monthtable
        st.write(df)
    with right_column:
        st_lottie(lottie_coding, height = 300, key ="coding")


# First Graph
    st.title("Plot Data :chart_with_upwards_trend:")
    st.markdown("###   ")
    st.subheader("Monthly Sale's from 2016 to 2022 chart")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    df = pd.read_csv('C:/Users/yakul/mutipage/data/month.csv')
   
    plt.figure(figsize=(12,6))
    plt.title('Sale Price History')
    plt.plot(df['Sale'])
    plt.xlabel('MONTHS', fontsize = 12)
    plt.ylabel('Sale Price PESO(₱)', fontsize = 12)
    st.pyplot()


# Second Graph
    st.subheader("Sale's vs Time chart with 12 months moving average")
    df = pd.read_csv('C:/Users/yakul/mutipage/data/month.csv')
    ma12 = df.Sale.rolling(12).mean()
    ma12
    plt.figure(figsize = (12,6))
    plt.plot(df.Sale)
    plt.plot(ma12)
    plt.title('Sale Price History')
    plt.xlabel('MONTHS', fontsize = 12)
    plt.ylabel('Sale Price PESO(₱)', fontsize = 12)
    plt.legend(['Sale','MA12'], loc = 'upper left')
    st.pyplot()


# Third Grapht

    st.subheader("Sale's vs Time chart with 12 months and 24 months moving average")
    df = pd.read_csv('C:/Users/yakul/mutipage/data/month.csv')
    ma12 = df.Sale.rolling(12).mean()
    ma12
    ma24 = df.Sale.rolling(24).mean()
    ma24
    plt.figure(figsize = (12,6))
    plt.plot(df.Sale)
    plt.plot(ma12)
    plt.plot(ma24)
    plt.title('Sale Price History')
    plt.xlabel('MONTHS', fontsize = 12)
    plt.ylabel('Sale Price PESO(₱)', fontsize = 12)
    plt.legend(['Sale','MA12', 'MA24'], loc = 'upper left')
    st.pyplot()
    



# Fourth Graph and load model
    import math
    
    st.subheader('Prediction Sale vs Original Sale')
    
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

    for i in range(12, len(train_data)):
        x_train.append(train_data[i-12:i, 0])
        y_train.append(train_data[i, 0])
    if i<= 12:
        print(x_train)
        print(y_train)
        print()

    x_train, y_train = np.array(x_train), np.array(y_train)



    x_train = np.reshape(x_train, (x_train.shape[0], 12, 1))
    x_train.shape

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

    test_data = scaled_data[training_data_len -12: , :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(12, len(test_data)):
        x_test.append(test_data[i -12:i,0])


    x_test = np.array(x_test)


    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)


    train = data[:training_data_len]
    prediction = data[training_data_len:]
    prediction['Predictions'] = predictions
    plt.figure(figsize=(12,6))
    plt.title('Sale Price History')
    plt.plot(data)
    plt.xlabel('MONTHS', fontsize = 12)
    plt.ylabel('Sale Price PESO(₱)', fontsize = 12)
    plt.plot(train['Sale'])
    plt.plot(prediction[[ 'Predictions']])
    plt.legend(['TEST','TRAIN','PREDICTION'], loc = 'upper left')
    st.pyplot()



    month = pd.read_csv('C:/Users/yakul/mutipage/data/year.csv', index_col='Date',parse_dates=True)
    df.index.frequency='MS'
    newdf = month.filter(['Sale'])
    last_12_years = newdf[-12:].values
    last_12_years_scaled = scaler.transform(last_12_years)
    X_test = [] 
    X_test.append(last_12_years)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    pred_sale = model.predict(X_test)
    pred_sale = scaler.inverse_transform(pred_sale)
    print(pred_sale)

# Prediction button

    st.subheader("Prediction for April 2022 Sale's")
    st.write(pred_sale)
