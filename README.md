# project2
#Loading_the_dataset
import pandas as pd
import numpy as np
pat=os.listdir('dataset/ps')
h=[]
for f in pat:
d=pd.read_csv('dataset/ps/'+f,index_col='Timestamp',parse_dates=True)
h.append(d)
data=pd.concat(h,axis=0)
#Data_Splitting
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
train, test, trainl, testl=train_test_split(
data['DL_bitrate'],
test_size=0.2,
random_state=41)
#Missing_Values_Removal_and_imputation
np.any(np.isnan(test))
#Finding the mean of the column having NaN
mean_value1=data['ServingCell_Lon'].mean()
mean_value2=data['ServingCell_Lat'].mean()
mean_value3=data['ServingCell_Distance'].mean()
data['ServingCell_Lon'].fillna(value=mean_value1, inplace=True)
data['ServingCell_Lat'].fillna(value=mean_value2, inplace=True)
data['ServingCell_Distance'].fillna(value=mean_value2, inplace=True)
train.dropna()
test.dropna()
data.dropna()
28
#Outlier_Detection and removal
data['zscore']=(data['DL_bitrate']-data['DL_bitrate'].mean())/data['DL_bitrate'].std()
data[data['zscore']>3]
dataz=data[(data['zscore']>-3)&(data['zscore']<3)]
dataz
#Feature_scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x_train= scaler.fit_transform(train)
x_train = pd.DataFrame(x_train)
x_test_scaled = scaler.fit_transform(test)
x_test = pd.DataFrame(x_test_scaled)
#Feature_selection
correlated_features = set()
correlation_matrix = data.corr()
for i in range(len(correlation_matrix .columns)):
for j in range(i):
if abs(correlation_matrix.iloc[i, j]) > 0.9:
colname = correlation_matrix.columns[i]
correlated_features.add(colname)
print(correlated_features)
#dropping_constant_features_using_variance_threshold
constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(train)
#importing required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
#train_the_model_using_KNN_for_Regression
for K in range(30):
K = K+1
29
model = neighbors.KNeighborsRegressor(n_neighbors = K)
model.fit(x_train, trainl)
#fit the model
pred=model.predict(x_test) #make prediction on test set
#calculating_the_rmse
rmse_val = [] #to store rmse values for different k
for K in range(30):
K = K+1
model = neighbors.KNeighborsRegressor(n_neighbors = K)
model.fit(x_train, trainl) #fit the model
pred=model.predict(x_test) #make prediction on test set
error = sqrt(mean_squared_error(testl,pred)) #calculate rmse
rmse_val.append(error) #store rmse values
print('RMSE value for k= ' , K , 'is:', error)
#plot_of_the_actual_values
plt.plot(t)
plt.xlabel('Time')
plt.ylabel('DL_Bandwidth')
plt.title('Actual Values')
#plot_of_predictions
plt.plot(pred)
plt.xlabel('Time')
plt.ylabel('DL_Bandwidth')
plt.title('KNN Predictions')
#Random_forest_for_Regression
from sklearn.ensemble import RandomForestRegressor
#training_the_model
regressor = RandomForestRegressor(n_estimators = 200, random_state = 0)
regressor.fit(x_train, trainl)
#making_prediction_on_test_set
prediction = regressor.predict(x_test)
#plot_of_the_actual_values
plt.plot(x_test)
plt.xlabel('Time')
plt.ylabel('DL_Bandwidth')
30
plt.title('Actual Values')
#plot_of_predictions
plt.plot(prediction)
plt.xlabel('Time')
plt.ylabel('DL_Bandwidth')
plt.title('Random Forest Predictions')
prediction = regressor.predict(x_test)
# Compute mean squared error for Random Forest for Regression
mse = mean_squared_error(testl, prediction)
# Print results
print(max(prediction))
print(mse)
print(pow(mse,0.5))
#ARIMA_Model
#importing_required_packages
from pmdarima import auto_arima
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")
#performing_adfuller_test
from statsmodels.tsa.stattools import adfuller
def adf_test(dataset):
dftest = adfuller(dataset, autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
print("\t",key, ": ", val)
#getting_the_order_for_training_model_for_arima
stepwise_fit = auto_arima(data['DL_bitrate'],
suppress_warnings=True)
stepwise_fit.summary()
#model_training_using_the_appropiate_order_for_p,q,d
31
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
model = sm.tsa.arima.ARIMA(trainl, order=(4,1,2))
modelf=model.fit()
modelf.summary()
#making_prediction_on_test_set
pred=modelf.predict(start=start,end=end,typ='levels').rename('ARIMA predictions')
#Plot_of_Actual_values
testl.plot(color='blue',figsize=(20,8),ylabel='DL_Bandwidth',xlabel='Time',title='Actual Data')
#Arima_predictions
import matplotlib.pyplot as plt
start=0
print(start)
end=len(test)-1
print(end)
pred=modelf.predict(start=start,end=end,typ='levels').rename('ARIMA predictions')
preds.plot(color='orange',figsize=(20,8),ylabel='DL_Bandwidth',xlabel='Time',title='ARIMA Predictions')
# Computing mean squared error for ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(pred,testl))
print(rmse)
#LSTM_Model
#importing_packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
model = Sequential()
model.add(LSTM(64, return_sequences = True,input_shape = [X_train.shape[1], X_train.shape[2]]))
model.add(Dropout(0.1))
model.add(LSTM(64))
model.add(Dropout(0.1))
model.add(Dense(units = 1))
#Compile model
32
model.compile(loss='mse', optimizer='adam')
#training_of_the_data
history=model.fit(X_train, y_train, epochs = 50,validation_split = 0.2, batch_size = 32,shuffle = False)
#predictions_on_test_set
def prediction(model):
prediction = model.predict(X_test)
prediction = scaler_y.inverse_transform(prediction)
return prediction
prediction_lstm = prediction(model)
#Plot_of_actual_values
plt.plot(y_test)
plt.rcParams["figure.figsize"] = (10,6)
plt.xlabel('Time')
plt.ylabel('DL_Bandwidth')
plt.title('Actual Data')
#LSTM_prediction_plot
def plot_future(prediction, y_test):
plt.figure(figsize=(10, 6))
range_future = len(prediction)
plt.plot(np.arange(range_future),np.array(prediction), )
plt.xlabel('Time')
plt.ylabel('DL_Bandwidth')
plt.title('LSTM Prediction')
plot_future(prediction_lstm, y_test)
#calculating_rmse_for_LSTM
errors = prediction_lstm - y_test
mse = np.square(errors).mean()
rmse = np.sqrt(mse)
