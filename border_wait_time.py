"""
Created on July 02 2019

@author: Peng Wang

Build predictive model using Facebook Prophet to predict hourly border crossing wait time at Peace Arch
Collected data from: Whatcom Council of Governments http://www.cascadegatewaydata.com/Crossing/
"""
from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
import pickle, glob, requests
from sklearn.metrics import mean_squared_error
from math import sqrt
import sys
import json
 
'''
PATH_DATA = r'./data/' 
FILE_MODEL = r'./models/forecast_model.pckl'
FILE_PREDICTION = './results/7_day_forecast.csv'
OUTPUT_IMG_FORECAST_COMP = './deployment/static/forecast_compenents.png'
OUTPUT_IMG_PREDICTION = './deployment/static/7_day_prediction.png'
FORECAST_DAYS = 7
'''
FILE_CONFIG = r'.\config.json'

# ---------------------------------------------------
# ------------- Data Cleansing   --------------------
# ---------------------------------------------------
def data_load():
    # Load border crossing wait time data from the folder
    data_files = glob.glob(config["file"]["PATH_DATA"] + "*.csv")
    data = pd.DataFrame()
    list_ = []
    for file_ in data_files:
        df = pd.read_csv(file_)
        list_.append(df)
    data = pd.concat(list_, ignore_index=True, sort=False)
    # Sort by date time
    data.sort_values(by='Group Starts', axis=0, inplace=True, ascending=True)
    # Rename columns 
    data.rename(columns={'Group Starts':'ds', 'Avg - Delay (Peace Arch)':'y'}, inplace=True)
    # Fill missing wait time with 0
    data['y'].fillna(0., inplace=True)
    data['ds'] = pd.to_datetime(data['ds'])
    return data

def get_test_start_date(data):
    test_end_date = datetime.date((data['ds'].max()))
    train_end_date = test_end_date - timedelta(days=config["number"]["FORECAST_DAYS"])
    test_start_date = train_end_date + timedelta(days=1)
    return test_start_date

def create_training(data, test_start_date):
    return data[data['ds'] < test_start_date]

def create_test(data, test_start_date):
    return data[data['ds'] >= test_start_date]        
            
# ---------------------------------------------------
# ------------- Train Prophet Model -----------------
# ---------------------------------------------------
def prophet_train(data_training):
    m = Prophet(seasonality_prior_scale=10, holidays_prior_scale=10)
    m.add_country_holidays(country_name='CA')
    #m.train_holiday_names
    m.fit(data_training)
    save_model(m)
    return m

def prophet_forecast(model):
    future = model.make_future_dataframe(freq='H', periods=24*config["number"]["FORECAST_DAYS"], include_history=True) # one week
    forecast = model.predict(future)
    #clips the forecasts so that no value is negative 
    forecast['yhat'] = forecast.yhat.clip_lower(0)
    forecast['yhat_lower'] = forecast.yhat.clip_lower(0)
    # forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    # Plot components
    prophet_plot_components(model, forecast)
    #Save forecasts to csv
    delay_forecast = forecast[['ds', 'yhat']][-24*config["number"]["FORECAST_DAYS"]:]
    delay_forecast.to_csv(config["file"]["FILE_PREDICTION"], float_format='%.f', index=None, \
                          header= ['Date Time','Expected Delay in Minutes'])
    return delay_forecast

# Plot components: historical, yearly, monthly, weekly, daily trends
def prophet_plot_components(m, forecast):
    fig = m.plot_components(forecast)
    fig.suptitle('Peace Arch border wait time trends')
    fig.savefig(config["file"]["OUTPUT_IMG_FORECAST_COMP"])
    print ("...Prophet plot of components is saved to  {}".format(config["file"]["OUTPUT_IMG_FORECAST_COMP"]))

# Plot hourly wait time forecast and compare to actual wait time
def plot_past_actual_and_forecast(pred_test):    
    test_start_date = datetime.date((pred_test['ds'].min()))
    test_end_date = datetime.date((pred_test['ds'].max()))
    pred_test.rename(columns={'ds':'Date Time', 'yhat':'Expected Delay in Minutes','y':'Delay in Minutes'}, inplace=True)    
    fig, axes = plt.subplots(config["number"]["FORECAST_DAYS"],1, figsize=(14,10),sharex=True)
    fig.suptitle('Peace Arch border wait time prediction {} to {}'.format(test_start_date, test_end_date))
    for i, ax in enumerate(axes):    
        curr_pred = pred_test.loc[pred_test['Date Time'].dt.date==test_start_date + timedelta(i)]
        ax.plot(curr_pred['Date Time'].dt.hour, curr_pred['Delay in Minutes'], 'o-')
        ax.plot(curr_pred['Date Time'].dt.hour, curr_pred['Expected Delay in Minutes'], 'o-')
        ax.set_xticks(curr_pred['Date Time'].dt.hour)
        ax.set_ylabel(test_start_date + timedelta(days=i))    
    axes[int(config["number"]["FORECAST_DAYS"]/2)].figure.text(0.05,0.5, "Delay (in Minutes)", \
        ha="center", va="center", rotation=90, fontsize='large')
    axes[0].legend(loc='upper left')
    plt.xlabel('Hour of the day', fontsize='large')
    fig.savefig(config["file"]["OUTPUT_IMG_PREDICTION"])

# Model Evaluation - RMSE
def model_eval(pred_test):
    rmse = sqrt(mean_squared_error(pred_test['Delay in Minutes'],pred_test['Expected Delay in Minutes']))
    print('RMSE = ', rmse)

def save_model(m):
    with open(config["file"]["FILE_MODEL"], 'wb') as fout:
        pickle.dump(m, fout)
        print ("...Model is saved to {}".format(config["file"]["FILE_MODEL"]))
        
def load_model():
    with open(config["file"]["FILE_MODEL"], 'rb') as fin:
        return pickle.load(fin)

def request_new_wait_time():  
    date_last_update = datetime.strptime(config['date']['LAST_DATE_BORDER_RECORD_UPDATE'], '%m/%d/%Y').date()
    days_to_update = (date.today() - date_last_update).days
    for i in range(1,days_to_update):     
        #Request new avg wait time since last update
        query_date = date_last_update + timedelta(days=i)
        query_date_str = query_date.strftime('%m/%d/%Y')
        # Update last date in config file
        config['date']['LAST_DATE_BORDER_RECORD_UPDATE'] = query_date_str
        query_date_filename = query_date.strftime('%Y%m%d')
        response = requests.get(config["url"]["datasource"]+"id="+config["border_id"]["PEACE_ARCH"]+"&start="+query_date_str+"&end="+query_date_str+"&data=avg-delay&dir=Southbound&lane=Car&tg=Hour&format=csv")
        
        if response.status_code == 200:
            raw_data = response.text
        else:
            print('An error has occurred.')
        
        with open(config["file"]["PATH_DATA"]+query_date_filename+'.csv', 'w', newline='') as csv_file:  
            csv_file.write(raw_data)
            print ("...Border crossing wait time record is saved to  {}".format(config["file"]["PATH_DATA"]+query_date_filename+'.csv'))
        csv_file.close()       
    
def request_yearly_wait_time(year = '2014'):        
    response = requests.get(config["url"]["datasource"]+"id="+config["border_id"]["PEACE_ARCH"]+"&start=01/01/"+year+"&end=12/31/"+year+"&data=avg-delay&dir=Southbound&lane=Car&tg=Hour&format=csv")
    
    if response.status_code == 200:
        raw_data = response.text
    else:
        print('An error has occurred.')    
    with open(FILE_CONFIG, 'w') as csv_file:  
        csv_file.write(raw_data)
    csv_file.close()
  
def update_config_file():
    with open(FILE_CONFIG, 'w') as config_file:  
        json.dump(config, config_file)    
    
def main():    
    data_training = []
    data_test = []
    test_start_date = date.today()
    predPast = False
    request_new_wait_time()    
    data = data_load()
    print ("...Data load is complete.")
    if (len(sys.argv) > 1 and sys.argv[1] == 'past'):
        predPast = True
        test_start_date = get_test_start_date(data)
        data_training = create_training(data,test_start_date)
        data_test = create_test(data,test_start_date)
    else:
        data_training = data    
        
    model = prophet_train(data_training)
    print ("...Model training is complete.")
    forecast = prophet_forecast(model)
    print ("...Model forecast is complete.")    
    if (predPast):
        pred_test = pd.merge_asof(data_test, forecast, on='ds')        
        plot_past_actual_and_forecast(pred_test)
        model_eval(pred_test)      
    # Update config file
    update_config_file()
    
# Load config variables    
with open(FILE_CONFIG) as json_config_file:
        config = json.load(json_config_file)
        
if __name__ == "__main__": 
    main()   