import pandas as pd
import numpy as np 
import pymysql
import datetime

sec = ('BOMPAT','BOMDEL')
ndo = 15
flt = ('585','319')

conn = pymysql.connect(host='localhost',port=3306,user='root',password='kunal',db='goair')

cursor = conn.cursor()

####Fetch bookings data
sql = """select Sector, FlgtNumb, Dept_date, Ndo, LF from bookings where Sector in %s and FlgtNumb in %s and Ndo in (%s,%s,%s,%s,%s);"""%(sec,flt,ndo,ndo+1,ndo+2,ndo+3,ndo+4)

cursor.execute(sql)
bookings = cursor.fetchall()
    
bookings = pd.DataFrame(list(bookings))
cols = ['Sector','Flt_Num','Dept_date','Ndo','LF']
bookings.columns = cols

###fetch price data
sql = """select * from price where Sector in %s and G8_Flt_Num in %s and Ndo = %s;"""%(sec,flt,ndo)

cursor.execute(sql)
price = cursor.fetchall()

price = pd.DataFrame(list(price))
cols = ['Sector','Ndo','Dept_date','G8_Flt_Num','G8_fare','Days_lowest','Timeband_lowest','6E','SG','UK','AI','9W','I5']
price.columns = cols

conn.close()

'''Actual bookings'''
bookings['Dept_date'] = pd.to_datetime(bookings['Dept_date'])
bookings['Dept_date'] = bookings['Dept_date'].apply(lambda x:x.date())

bookings.sort_values(by = ['Sector','Flt_Num','Dept_date','Ndo'],axis = 0,inplace=True)
data = bookings.values.tolist()

for i in range(len(data)-1):
    currRow = data[i]
    nextRow = data[i+1]
    
    if currRow[0] == nextRow[0] and currRow[1] == nextRow[1] and currRow[2] == nextRow[2]:
        for column in range(4, len(currRow)):
            data[i][column] = data[i][column] - data[i+1][column]

df1 = pd.DataFrame(data, columns=bookings.columns)
df1 = df1[df1['Ndo'] < ndo+4]
data = pd.pivot_table(df1,index=['Sector','Flt_Num','Dept_date'],values=['LF'],columns=['Ndo'],fill_value=0)
data.columns = [x for x in range(ndo,ndo+4)]
data.reset_index(drop=False,inplace=True)
    
data['DOW'] = data['Dept_date'].apply(lambda x:x.weekday())
just_dummies = pd.get_dummies(data['DOW'],prefix='dow')
data = pd.concat([data, just_dummies], axis=1)
data = data.drop(['DOW'],axis=1)
data['Dept_date']=data['Dept_date'].astype(str)
data['key'] = data['Sector']+data['Flt_Num']+data['Dept_date']

price = pd.pivot_table(price,index=['Sector','G8_Flt_Num','Dept_date'],values=['G8_fare', 'Days_lowest',
       'Timeband_lowest', '6E', 'SG', 'UK', 'AI', '9W', 'I5'],aggfunc='min')
price.reset_index(inplace=True)
price['Cap_date'] = price['Dept_date']-datetime.timedelta(days=ndo) 
price['dow_bk'] = price['Cap_date'].apply(lambda x:x.weekday())
just_dummies = pd.get_dummies(price['dow_bk'],prefix='dow_bk')
price = pd.concat([price, just_dummies], axis=1)
price = price.drop(['dow_bk','Cap_date'],axis=1)
price['Dept_date']=price['Dept_date'].astype(str)
price['key']=price['Sector']+price['G8_Flt_Num']+price['Dept_date']
price.drop(['Sector','G8_Flt_Num','Dept_date'],inplace=True,axis=1)

data = data.merge(price,on='key',how='left')
data = data[np.isfinite(data['6E'])]
just_dummies = pd.get_dummies(data['Sector'],prefix='sec')
data = pd.concat([data, just_dummies], axis=1)
just_dummies = pd.get_dummies(data['Flt_Num'],prefix='flt')
data = pd.concat([data, just_dummies], axis=1)

#data['month'] = data['Dept_date'].apply(lambda x:x.month)
#just_dummies = pd.get_dummies(data['month'],prefix='month')
#data = pd.concat([data, just_dummies], axis=1)

data = data.drop(['key','Sector','Flt_Num','Dept_date'],axis=1)
for col in data.columns:
    data[col][data[col] < 0] = 0


scrapped = pd.read_csv(r'C:\Users\kunald\Desktop\merge files\Scrapped.csv')
scrapped = scrapped.iloc[:,0:16]
scrapped['Sector'] = scrapped['Origin']+scrapped['Destination']
scrapped = scrapped[scrapped['Sector']=='BOMPAT']
scrapped['key']=scrapped['Dept_Date']+scrapped['Airline']+scrapped['Flight_Code']
scrapped.drop_duplicates(['key'],inplace=True)
fltcount = pd.pivot_table(scrapped,index=['Dept_Date'],values=['Flight_Code'],aggfunc='count')
fltcount.reset_index(inplace=True)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from math import sqrt
from numpy import concatenate
# ensure all data is float
values = data.values
values = values.astype('float32')
#pyplot.figure()
#pyplot.plot(values[:,0])

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
k = len(data.columns)
reframed.drop(reframed.columns[k+1:], axis=1, inplace=True)
n = len(reframed)*0.90

# split into train and test sets
values = reframed.values
train = values[:int(n), :]
test = values[int(n):, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=1000, batch_size=10, validation_data=(test_X, test_y), verbose=2, shuffle=False)

#------model 2-------------------

model = Sequential()
model.add(Dropout(0.01,input_shape=(1,k)))
model.add(Dense(15))
model.add(LSTM(5))
#model.add(LSTM(25, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=1000, batch_size=15, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

pyplot.plot(inv_yhat,label='yhat')
pyplot.plot(inv_y,label='y')
pyplot.legend()
pyplot.show()


