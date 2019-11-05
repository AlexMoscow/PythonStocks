#!/usr/bin/env python
# coding: utf-8

# In[47]:


from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tulipy as ti
get_ipython().run_line_magic('matplotlib', 'inline')
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
df = joblib.load("../Documents/MSFT.pkl")
df['test'] = df['open']
df.test = df.test.shift(-1) #сдвигаю вниз
y = df['open'].values
z = df['close'].values
high = df['high'].values
low = df['low'].values
#df = df.dropna()           #удаляю первую строку
new_o = df.loc[:,['open']] #фильтр
ewma = pd.Series.ewm

'''cal = calendar()
holidays = cal.holidays(start='2007-04-25', end='2007-10-07')
df['holiday'] = df['datetime'].isin(holidays)'''

#скользящая средняя
def moving_average(new_o, n): 
    MA = pd.Series(new_o['open'].rolling(window=n).mean())
    return MA
#print(moving_average(new_o['open'], 5)

#Экспоненциальная скользящая средняя
def exponential_moving_average(new_o, n):
    EMA = pd.Series(new_o['open'].ewm(span=n, min_periods=n).mean())
    return EMA

#разбивка на тренировочное и тестовое мн-ва
NumberOfElements = len(df)
TrainingSize = int(NumberOfElements * 0.7)
TrainingData = df[0:TrainingSize]
TestData = df[TrainingSize:NumberOfElements]

'''def StartARIMAForecasting(Actual, P, D, Q):
    model = ARIMA(Actual, order=(P, D, Q))
    model_fit = model.fit(disp=0)
    prediction = model_fit.forecast()[0]
    return prediction
predicted = StartARIMAForecasting(new_o, 1,1,0)
print('Predicted=%f' % (predicted))'''

#RSI индикатор
def RSI(series, period):
    delta = series.diff()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) #сумма средних увеличений
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) #сумма средних уменьшений
    d = d.drop(d.index[:(period-1)])
    rs = ewma(u, span=period-1).mean() / ewma(d, span=period-1).mean()
    return 100 - 100 / (1 + rs)

#Стохастик индикатор
def stochastic_oscillator(df, n):
    SOk = pd.Series((df['close'] - df['low']) / (df['high'] - df['low']))*100
    SOd = pd.Series(SOk.ewm(span=n, min_periods=n).mean())
    return SOd

new_o['RSI'] = RSI(df.close, 14) 
df['Stochastic'] = stochastic_oscillator(df, 5)
new_o['EMA'] = exponential_moving_average(new_o, 5)
new_o['MA'] = moving_average(new_o, 5)
#pred = new_o['MA'].values
#MSE = mean_squared_error(y, pred)
df['MA'] = new_o['MA']
df['EMA'] = new_o['EMA']
df['RSI'] = new_o['RSI']
df = df.loc['2007-04-25':'2007-10-07']  #фильтр для детализации
plt.plot(df.index,df['test'], label='test')
plt.plot(df.index,df['MA'],label='MA', color = 'red')
plt.plot(df.index,df['EMA'],label='EMA', color = 'green')
plt.legend()
plt.title('Microsoft')
plt.ylabel('Price($)')
plt.show()
df.plot(y=['RSI'])
df.plot(y=['Stochastic'])

#Tulip Indicators
stoch_k, stoch_d = ti.stoch(high, low, z, 5, 3, 3)
sma = ti.sma(y, 5)
rsi = ti.rsi(z, 14)
ema = ti.ema(y,5)
print('MA: ', sma)
print('EMA: ',ema)
print('RSI: ',rsi)
print('Stochastic: ',stoch_d)


# In[44]:


new_o


# In[29]:


df


# In[55]:


df.plot(y=['RSI'])


# 

# In[ ]:




