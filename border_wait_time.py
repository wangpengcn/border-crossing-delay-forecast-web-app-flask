"""
Created on Tue Sep  4 15:50:07 2018

@author: Peng Wang

Build predictive model using Facebook Prophet to predict hourly border crossing wait time at Peace Arch
Collected data from:
- Border wait time: Whatcom Council of Governments http://www.cascadegatewaydata.com/Crossing/

"""
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning, )

from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
import pickle, glob, requests
from sklearn.metrics import mean_squared_error
from math import sqrt

 
PATH_DATA = r'./data/' 
PATH_RESULTS = r'./results/' 
FILE_MODEL = r'./models/forecast_model.pckl'
FILE_PREDICTION = PATH_RESULTS + '7_day_forecast.csv'
OUTPUT_IMG_FORECAST_COMP = './deployment/static/forecast_compenents.png'
OUTPUT_IMG_PREDICTION = './deployment/static/7_day_prediction.png'
FORECAST_DAYS = 7

# ---------------------------------------------------
# ------------- Data Exploration --------------------
# ---------------------------------------------------
# Load in border crossing wait time data
data_files = glob.glob(PATH_DATA + "*.csv")
data = pd.DataFrame()
list_ = []
for file_ in data_files:
    df = pd.read_csv(file_)
    list_.append(df)
data = pd.concat(list_, ignore_index=True, sort=False)
data.sort_values(by='Group Starts', axis=0, inplace=True, ascending=True)
# Rename columns 
data.rename(columns={'Group Starts':'Date Time', 'Avg - Delay (Peace Arch)':'Delay'}, inplace=True)
# Fill missing wait time with 0
data['Delay'].fillna(0., inplace=True)
data['Date Time'] = pd.to_datetime(data['Date Time'])
test_end_date = datetime.date((data['Date Time'].max()))
train_end_date = test_end_date - timedelta(days=FORECAST_DAYS)
test_start_date = date.today()#train_end_date + timedelta(days=1)

data.columns = ['ds', 'y']
data_training = data[data['ds'] < test_start_date]
data_test = data[data['ds'] >= test_start_date]

# ---------------------------------------------------
# ------------- Train Prophet Model -----------------
# ---------------------------------------------------
m = Prophet(seasonality_prior_scale=10, holidays_prior_scale=10)
m.add_country_holidays(country_name='CA')
#m.train_holiday_names
m.fit(data_training)
future = m.make_future_dataframe(freq='H', periods=24*FORECAST_DAYS, include_history=True) # one week
forecast = m.predict(future)
#clips the forecasts so that no value is negative 
forecast['yhat'] = forecast.yhat.clip_lower(0)
forecast['yhat_lower'] = forecast.yhat.clip_lower(0)
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
#Save forecasts to csv
delay_forecast = forecast[['ds', 'yhat']][-24*FORECAST_DAYS:]
delay_forecast.to_csv(FILE_PREDICTION, float_format='%.f', index=None, \
                      header= ['Date Time','Expected Delay in Minutes'])
# ---------------------------------------------------
# ------------- Plot Forecasts ---------------------
# ---------------------------------------------------
# Plot components: historical, yearly, monthly, weekly, daily trends
fig = m.plot_components(forecast)
fig.suptitle('Peace Arch border wait time trends')
fig.savefig(OUTPUT_IMG_FORECAST_COMP)

# Plot hourly wait time forecast and compare to actual wait time
pred_test = pd.merge_asof(data_test, delay_forecast, on='ds')
# Rename column
pred_test.rename(columns={'ds':'Date Time','y':'Delay in Minutes', 'yhat':'Expected Delay in Minutes'}, inplace=True)
fig, axes = plt.subplots(FORECAST_DAYS,1, figsize=(14,10),sharex=True)
fig.suptitle('Peace Arch border wait time prediction {} to {}'.format(test_start_date, test_end_date))
for i, ax in enumerate(axes):    
    curr_pred = pred_test.loc[pred_test['Date Time'].dt.date==test_start_date + timedelta(i)]
    ax.plot(curr_pred['Date Time'].dt.hour, curr_pred['Delay in Minutes'], 'o-')
    ax.plot(curr_pred['Date Time'].dt.hour, curr_pred['Expected Delay in Minutes'], 'o-')
    ax.set_xticks(curr_pred['Date Time'].dt.hour)
    ax.set_ylabel(test_start_date + timedelta(days=i))    
axes[int(FORECAST_DAYS/2)].figure.text(0.05,0.5, "Delay (in Minutes)", \
    ha="center", va="center", rotation=90, fontsize='large')
axes[0].legend(loc='upper left')
plt.xlabel('Hour of the day', fontsize='large')
fig.savefig(OUTPUT_IMG_PREDICTION)

# Model Evaluation - RMSE
rmse = sqrt(mean_squared_error(pred_test['Delay in Minutes'],pred_test['Expected Delay in Minutes']))
print('RMSE = ', rmse)

def save_model():
    with open(FILE_MODEL, 'wb') as fout:
        pickle.dump(m, fout)
        
def load_model():
    with open(FILE_MODEL, 'rb') as fin:
        return pickle.load(fin)

def request_yesterday_wait_time():    
    #Request yesterady avg wait time
    query_date = date.today()-timedelta(days=1)
    query_date_str = query_date.strftime('%m/%d/%Y')
    query_date_filename = query_date.strftime('%Y%m%d')
    response = requests.get("http://www.cascadegatewaydata.com/Crossing/?id=134&start="+query_date_str+"&end="+query_date_str+"&data=avg-delay&dir=Southbound&lane=Car&tg=Hour&format=csv")
    
    if response.status_code == 200:
        raw_data = response.text
    else:
        print('An error has occurred.')
    
    with open(PATH_DATA+query_date_filename+'.csv', 'w', newline='') as csv_file:  
        csv_file.write(raw_data)
    csv_file.close()

def request_yearly_wait_time(year = '2014'):        
    response = requests.get("http://www.cascadegatewaydata.com/Crossing/?id=134&start=01/01/"+year+"&end=12/31/"+year+"&data=avg-delay&dir=Southbound&lane=Car&tg=Hour&format=csv")
    
    if response.status_code == 200:
        raw_data = response.text
    else:
        print('An error has occurred.')    
    with open(PATH_DATA+year+'.csv', 'w', newline='') as csv_file:  
        csv_file.write(raw_data)
    csv_file.close()