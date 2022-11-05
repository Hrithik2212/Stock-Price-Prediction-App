import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM , Dropout , Dense
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  mean_absolute_percentage_error as mape 
from datetime import datetime as dt
from datetime import date
import pandas_datareader as pdr
import warnings
warnings.simplefilter(action='ignore')
from tensorflow.keras.layers import Bidirectional



class Data_API:
    def __init__(self)->None:
        self.ss = StandardScaler()

        '''    ## Test
        # Test 1 : Call data 
        print("\nTest 1\n")
        data = self.data_call("^BSESN")
        print(data.head())

        # Test 2 : Describe the Data 
        print("\nTest 2\n")
        description = self.describe()
        print(description)

        # Test 3 : Scale the Data 
        print("\nTest 3\n")
        s_d = self.scaller(data)
        print(f"Original Data , \n{data[:5]}")
        print(f"Scalled Data , \n{s_d[:5]}")

        # Test 4 : Iverse Scale the Scalled Data
        print("\nTest 4\n")
        i_s_d = self.inverse_scaller(s_d)
        print(f"Rescalled Data ,\n{i_s_d[:5]}")   
        
        '''

    def data_call(self,ticker,end=dt.now(),start='2019-01-01'):
        try : 
            self.df = pdr.DataReader(ticker,"yahoo",start,end)
            return self.df["Close"]
        except :
            print("OOPS , it seems something has went wrong")
            return None

    def scaller(self, target):
        data = np.array(target)
        data = data.reshape(-1,1)
        scalled_data = self.ss.fit_transform(data)
        return scalled_data

    def inverse_scaller(self, data):
        return self.ss.inverse_transform(data)
    
    def describe(self):
        return self.df.describe()




def get_keras_format_series(series):
    """
    Convert a series to a numpy array of shape 
    [n_samples, time_steps, features]
    """
    
    series = np.array(series)
    return series.reshape(series.shape[0], series.shape[1], 1) 


class Data_Formatter:
    def __init__(self)->None:
        pass

    def get_train(self , series ,splits =50):

        '''
        Converts the previous data to Tensflow freindly format 
        to train the data  
        '''
        X , y = [] , []
        if type(series) != type(np.array([])):
            series = np.array(series)
            
        for i in range(0,series.shape[0]-splits):
            X.append(series[i:i+splits])
            y.append(series[i+splits])

        X = get_keras_format_series(X)
        y = np.array(y).reshape(-1,1)
        X_init = series[-splits:]

        return X , y , X_init


class BiLSTM:
    '''
    Biderction LSTM model 
    '''
    def __init__(self , input_shape = (50,1,1)):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(64 ,input_shape=(input_shape[0],input_shape[1]),activation='relu',return_sequences=True)))
        self.model.add(Dropout(0.1))
        self.model.add(Bidirectional(LSTM(64 , activation = "relu")))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error' , optimizer = "adam")
        self.model.build(input_shape=(50,1,1))
        self.model.summary()

    def train(self, X ,y ):
        self.X = X.copy()
        self.model.fit(X,y)

    def predict_future(self , X_init , n_steps ):
        
        X_init = X_init.copy().reshape(1,-1,1)
        preds = []
        for _ in range(n_steps):
            pred = self.model.predict(X_init)
            preds.append(pred)
            X_init[:,:-1,:] = X_init[:,1:,:] 
            X_init[:,-1,:] = pred 
    
        preds = np.array(preds).reshape(-1,1)
    
        return preds

    def test_plot(self, dates , y , X_init , future , scaler:StandardScaler):
                
        forecast = pd.date_range(start = str(dt.now())[:10] , periods=future ,freq='D')
        
        preds = self.predict_future(X_init , future)
        preds = scaler.inverse_transform(preds)
        plt.figure(figsize=(16,8))
        plt.plot(dates , y)
        plt.plot(forecast, preds , 'o-')

## Testing ## 

bse = Data_API()
