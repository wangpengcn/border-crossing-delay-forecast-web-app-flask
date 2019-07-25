"""
Created on July 02 2019

@author: Peng Wang

Build predictive model using Facebook Prophet to predict hourly border crossing wait time at Peace Arch
Collected data from: Whatcom Council of Governments http://www.cascadegatewaydata.com/Crossing/
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
from time import time
import pickle, glob, requests
from math import sqrt
import json, os
import holidays
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV,TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# Turn interactive plotting off
plt.ioff()

FILE_CONFIG = r'./config.json'

# Load border crossing wait time data from the folder
def data_load():    
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
    data.rename(columns={'Group Starts':'Date_time', 'Avg - Delay (Peace Arch)':'Delay'}, inplace=True)
    # Fill missing wait time with 0
    data['Delay'].fillna(0., inplace=True)
    # Add forecast datetime - future 7 days. If this is done separately  
    # after model is trained, dummy holiday features will have to be added
    # to the forecast dataset.
    # Future forecast starts "today 00:00:00"
    forecast_start_datetime = datetime(date.today().year,date.today().month,date.today().day)
    # Create a list of future timestamps by adding one more hour 
    forecast_dt_list = [forecast_start_datetime+ timedelta(hours=1*x) for x in range(0, 24*config["number"]["FORECAST_DAYS"])]
    # Convert it to dataframe
    forecast_dt_df = pd.DataFrame(forecast_dt_list,columns=['Date_time'])
    # Add empty column Delay  
    forecast_dt_df['Delay']=''
    # Concatenate with training dataframe
    data = pd.concat([data,forecast_dt_df], axis=0)
    # Convert Date_time column from string to datetime
    data['Date_time'] = pd.to_datetime(data['Date_time'])
    # Add more datetime related features
    data = add_datetime_features(data)  
    
    return data

# Add datetime related features
def add_datetime_features(data):
    data['Date'] = data['Date_time'].dt.date
    data['HourOfDay'] = data['Date_time'].dt.hour        
    data['Year'] = data['Date_time'].dt.year
    data['Month'] = data['Date_time'].dt.month
    data['DayOfMonth'] = data['Date_time'].dt.day 
    data['DayOfWeek'] = data['Date_time'].dt.dayofweek 
    # Get Canadian - BC holidays
    ca_holidays = holidays.CountryHoliday('CA', prov='BC', state=None)
    # Check each date what Canadian holiday it is
    data['Holiday_CA'] = [ca_holidays.get(x) for x in data['Date_time']]
    # Treat Observed holiday same as regular
    data['Holiday_CA'] = pd.Series(data['Holiday_CA']).str.replace(" \(Observed\)", "")
    # Convert holiday columns
    data = pd.get_dummies(data, columns=['Holiday_CA'])
    # Get US - WA holidays
    us_holidays = holidays.CountryHoliday('US', prov=None, state='WA')
    data['Holiday_US'] = [us_holidays.get(x) for x in data['Date_time']]
    data['Holiday_US'] = pd.Series(data['Holiday_US']).str.replace(" \(Observed\)", "")
    data = pd.get_dummies(data, columns=['Holiday_US'])
    
    return data

# Get test start date (because we split our data to training and test/validation)
def get_test_start_date(data):
    test_end_date = datetime.date((data['Date_time'].max()))
    train_end_date = test_end_date - timedelta(days=config["number"]["FORECAST_DAYS"])
    test_start_date = train_end_date + timedelta(days=1)
    return test_start_date  
            
# 1. Train XGBoost model
# 2. Get prediction on test data
# 3. Plot predictions (Expected Delay) vs labels (Delay)
# 4. Model evaluation by computing RMSE
def train_model(train_data):    
    # Drop Date_time and Date columns (no use anymore), and Delay (label)
    train_X = train_data.drop(['Date_time', 'Date', 'Delay'], axis=1)
    train_y = train_data['Delay']
    
    #-------- Build XGBoost model BEGIN --------------
    xgb_regressor = xgb.XGBRegressor(random_state=29, n_jobs=-1)
    # Model parameters
    xgb_parameters = {
                      'n_estimators': [80, 100, 120],
                      'learning_rate': [0.01, 0.1, 0.5],
                      'gamma': [0, 0.01, 0.1],
                      'reg_lambda': [0.5, 1],
                      'max_depth': [3, 5, 10], # Max depth of tree. Deeper -> overfitting
                      'subsample': [0.5, 1.0], # Subsample ratio of training instances
                      'colsample_bytree': [0.5, 0.7, 1], # Subsample ratio of columns of each tree
                      'seed':[0]
                      }
    n_iter = 48
    tscv = TimeSeriesSplit(n_splits=4)
    xgb_grid_search = RandomizedSearchCV(xgb_regressor, 
                                  xgb_parameters, 
                                  n_iter = n_iter, 
                                  cv=tscv,
                                  scoring = 'neg_mean_squared_error',
                                  verbose=1,                                  
                                  n_jobs=-1,                                  
                                  random_state= 50)
    start = time() # start time
    xgb_grid_fit = xgb_grid_search.fit(train_X, train_y.values)
    model = xgb_grid_search.best_estimator_
    end = time() # end time
    # Calculate training time
    xgb_time = (end-start)/60.
    print('---------------------------------------')
    print('Took {0:.2f} minutes to find optimized parameters for XGB model'.format(xgb_time))
    print('Best parameters for XGB model: {}'.format(xgb_grid_fit.best_params_))
    print('RMSE for XGB model: {}'.format(sqrt(-xgb_grid_fit.best_score_)))
    print('---------------------------------------')    
    #-------- Build XGBoost model END --------------
    
    # Save the model to file
    save_model(model)
    save_json_file(config['file']['FILE_MODEL_PARAM'],xgb_grid_fit.best_params_)    
    
    return model

def eval_model(model, test_data):
    test_start_date = test_data['Date'].min()
    # Extract test/validation data
    test_data = test_data[test_data['Date'] >= test_start_date]
    test_data.reset_index(inplace=True, drop=True)     
    test_X = test_data.drop(['Date_time', 'Date', 'Delay'], axis=1)
    #test_y = test_data['Delay']    
    # Get prediction on test data from the best iteration (default is the last iteration)
    test_probs = model.predict(test_X)
    # Concatenate prediction with test data
    pred_test = pd.concat([test_data, pd.DataFrame({'Expected Delay':test_probs})], axis=1)
    # Plot predictions (Expected Delay) vs labels (Delay)
    plot_test_pred_label(pred_test, test_start_date)
    # Model evaluation by computing RMSE
    model_eval(pred_test)

# Plot hourly wait time forecast and compare to actual wait time
def plot_test_pred_label(pred_test, test_start_date):
    fig, axes = plt.subplots(config["number"]["FORECAST_DAYS"],1, figsize=(14,10),sharex=True)
    fig.suptitle('Border wait time prediction {} to {}'.format(test_start_date, test_start_date+timedelta(days=config["number"]["FORECAST_DAYS"]-1)))
    for i, ax in enumerate(axes):    
        curr_pred = pred_test.loc[pred_test['Date_time'].dt.date==test_start_date + timedelta(i)]
        ax.plot(curr_pred['Date_time'].dt.hour, curr_pred['Delay'], 'o-')
        ax.plot(curr_pred['Date_time'].dt.hour, curr_pred['Expected Delay'], 'o-')
        ax.set_xticks(curr_pred['Date_time'].dt.hour)
        ax.set_ylabel(test_start_date + timedelta(days=i))    
    axes[int(config["number"]["FORECAST_DAYS"]/2)].figure.text(0.05,0.5, "Delay (in Minutes)", \
        ha="center", va="center", rotation=90, fontsize='large')
    axes[0].legend(loc='upper left')
    plt.xlabel('Hour of the day', fontsize='large')
    fig.savefig(config["file"]["OUTPUT_IMG_PREDICTION"])

# Test evaluation - RMSE
def model_eval(pred_test):
    test_start_date = pred_test['Date'].min().strftime('%Y-%m-%d')
    test_end_date = pred_test['Date'].max().strftime('%Y-%m-%d')
    rmse = sqrt(mean_squared_error(pred_test['Delay'],pred_test['Expected Delay']))
    print('...Test evaluation ({} to {}): RMSE = {:.2f}'.format(test_start_date, test_end_date, rmse))
    rmse_df = pd.DataFrame([[test_start_date, test_end_date, rmse]], columns=['Start Date', 'End Date', 'RMSE'])
    
    if (os.path.isfile(config["file"]["RMSE_RESULTS"])):
        rmse_results = pd.read_csv(config["file"]["RMSE_RESULTS"])
        rmse_df = pd.concat([rmse_results,rmse_df])    
    rmse_df.sort_values(by='Start Date', ascending=False, inplace=True)
    rmse_df.to_csv(config["file"]["RMSE_RESULTS"],float_format='%.2f',index=None)
    print ("...Future forecast is saved to {}".format(config["file"]["RMSE_RESULTS"]))

def delay_forecast(model, data_forecast): 
    result = pd.DataFrame(data_forecast['Date_time'],columns=['Date_time'])
    data_forecast = data_forecast.drop(['Delay','Date_time','Date'], axis=1)
    forecast_y = model.predict(data_forecast)
    future_forecast = pd.concat([data_forecast, pd.DataFrame({'Expected Delay in Minutes':forecast_y})], axis=1)
    result['Delay'] = forecast_y
    result.rename(columns={'Delay':'Expected Delay in Minutes'}, inplace=True)
    result.to_csv(config["file"]["FILE_PREDICTION"], float_format='%.f', index=None)
    return future_forecast

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
    new_data = False
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
            print('Cannot download new data file.')
        
        if raw_data:            
            with open(config["file"]["PATH_DATA"]+query_date_filename+'.csv', 'w', newline='') as csv_file:  
                csv_file.write(raw_data)
                print ("...Border crossing wait time record is saved to  {}".format(config["file"]["PATH_DATA"]+query_date_filename+'.csv'))
            csv_file.close()  
            # Update config file
            save_json_file(FILE_CONFIG,config)
            new_data = True
            
    return new_data
    
def request_yearly_wait_time(year = '2014'):        
    response = requests.get(config["url"]["datasource"]+"id="+config["border_id"]["PEACE_ARCH"]+"&start=01/01/"+year+"&end=12/31/"+year+"&data=avg-delay&dir=Southbound&lane=Car&tg=Hour&format=csv")
    
    if response.status_code == 200:
        raw_data = response.text
    else:
        print('Cannot download data file.')    
    
    if response.status_code == 200:
            raw_data = response.text
    else:
        print('Cannot download new data file.')
    
    if raw_data:            
        with open(config["file"]["PATH_DATA"]+year+'.csv', 'w', newline='') as csv_file:  
            csv_file.write(raw_data)
            print ("...Border crossing wait time record is saved to  {}".format(config["file"]["PATH_DATA"]+year+'.csv'))
        csv_file.close() 
          
def save_json_file(filename, obj):
    with open(filename, 'w') as json_file:  
        json.dump(obj, json_file) 
        
def read_json_file(filename):
    with open(filename) as json_file:
        obj = json.load(json_file)
    return obj
        
def main():
    # Download yearly wait time data since 2014, if not exist
    for i in range(2014,date.today().year):
        if (os.path.isfile(config["file"]["PATH_DATA"]+str(i)+'.csv')==False):
            request_yearly_wait_time(str(i))
    # Forecast start date    
    forecast_start_date = date.today()
    # Download new border crossing wait time records since last download
    new_data = request_new_wait_time() 
    # Load data files
    data = data_load()
    print ("...Data load is complete.")
    # Get training and test/eval data
    data_train_test = data[data['Date'] < forecast_start_date]
    # Get test/eval start date        
    test_start_date = get_test_start_date(data_train_test)
    train_data = data_train_test[data_train_test['Date'] < test_start_date]
    test_data = data_train_test[data_train_test['Date'] >= test_start_date]
    if new_data:  
       # Train model
        model = train_model(train_data) 
    else:
        model = load_model()
        print ("...Model is loaded.")   
    # Evaluate model performance using test data
    eval_model(model, test_data)
    print ("...Model training is complete.") 
      
    data_forecast = data[data['Date'] >= forecast_start_date]
    delay_forecast(model,data_forecast)      
    
# Load config variables    
config = read_json_file(FILE_CONFIG)
        
if __name__ == "__main__": 
    main()   