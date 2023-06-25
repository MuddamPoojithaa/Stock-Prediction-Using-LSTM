#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import yfinance as yf

import warnings
warnings.filterwarnings('ignore')



# In[22]:


stock_data = pd.read_csv("C:/Users/dell/Downloads/Google_Stock_Price_Train.csv")
test_data = pd.read_csv("C:/Users/dell/Downloads/Google_Stock_Price_Test.csv")


# In[23]:


stock_data.head()


# In[24]:


stock_data.tail()


# In[25]:


stock_data.info()


# In[26]:


stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data = stock_data.sort_values(by=['Date'], ascending=True).reset_index()


# In[27]:


stock_data.head()


# In[28]:


stock_data.tail()


# In[29]:


plt.figure(figsize=(18, 8))
plt.plot(stock_data['Open'])
plt.title("Google Stock Prices")
plt.xlabel("Time (oldest -> latest)")
plt.ylabel("Stock Opening Price")
plt.show()


# In[30]:


plt.figure(figsize=(18, 8))
plt.plot(stock_data['High'])
plt.title("Google Stock Prices")
plt.xlabel("Time (oldest-> latest)")
plt.ylabel("Stock Hightest Points")
plt.show()


# In[31]:


plt.figure(figsize=(18, 8))
plt.plot(stock_data['Low'])
plt.title("Google Stock Prices")
plt.xlabel("Time (oldest -> latest)")
plt.ylabel("Stock Lowest Points")
plt.show()


# In[32]:


plt.figure(figsize=(18, 8))
plt.plot(stock_data['Volume'])
plt.title("Volume of stocks sold")
plt.xlabel("Time (oldest-> latest)")
plt.ylabel("Volume of stocks traded")
plt.show()


# In[33]:


input_feature = stock_data[['Open', 'High', 'Low', 'Volume', 'Close']]
input_data = input_feature.values


# In[34]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
input_data[:,:] = scaler.fit_transform(input_data[:,:])


# In[35]:


lookback=50
total_size=len(stock_data)
X=[]
y=[]
for i in range(0, total_size-lookback): # loop data set with margin 50 as we use 50 days data for prediction
    t=[]
    for j in range(0, lookback): # loop for 50 days
        current_index = i+j
        t.append(input_data[current_index, :]) # get data margin from 50 days with marging i
    X.append(t)
    y.append(input_data[lookback+i, 4])


# In[36]:


test_size=100 # 100 days for testing data
X, y= np.array(X), np.array(y)
X_test = X[:test_size]
Y_test = y[:test_size]

X_work = X[test_size:]
y_work = y[test_size:]

validate_size = 10

X_valid = X[:validate_size]
y_valid = y[:validate_size]
X_train = X[validate_size:]
y_train = y[validate_size:]


# In[37]:


X_train = X_train.reshape(X_train.shape[0], lookback, 5)
X_valid = X_valid.reshape(X_valid.shape[0], lookback, 5)
X_test = X_test.reshape(X_test.shape[0], lookback, 5)
print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)


# In[38]:


get_ipython().system('pip install keras')


# In[39]:



get_ipython().system('pip install keras')


# In[40]:


get_ipython().system('pip install tensorflow')


# In[10]:


ticker = 'TSLA'
start_d = '2020-01-01'
end_d = '2023-06-19'
data=yf.download(ticker,start=start_d,end=end_d)
print(data)



# In[14]:


stock_data=pd.DataFrame(data)
stock_data.head()


# In[17]:




stock_data['Date']=pd.to_datetime(df.index)
stock_data.head()


# In[18]:



import plotly.graph_objects as go
fig=go.Figure(data=[go.Candlestick(x=df['Date'],
                                  open=df['Open'],
                                  high=df['High'],
                                  low=df['Low'],
                                  close=df['Close'])])
fig.update_layout(title='Stock Price Chart of TESLA',
                 yaxis_title='prices ($)',
                 xaxis_rangeslider_visible=False)
fig.show()


# In[20]:



df.reset_index(drop=True,inplace=True)


# In[23]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
x=stock_data[['Open','Close','High','Low','Adj Close']]
y=stock_data['Close']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
rf=RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
print('mean squared error',mse)


# In[22]:


import numpy as np
n_data=np.array([[250.970001,259.679993,261.570007,258.950012,263.600006]])
pred_price=rf.predict(n_data)
print('PREDICTED STOCK PRICE IS:',pred_price[0])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




