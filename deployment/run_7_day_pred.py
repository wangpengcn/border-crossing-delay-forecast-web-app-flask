# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 15:50:02 2019

@author: wangp
"""

from flask import Flask, request, render_template, session
import pandas as pd
from datetime import datetime, timedelta, date
from wtforms.fields.html5 import DateField
from wtforms_components import DateRange
from flask_wtf import Form
from flask_bootstrap import Bootstrap
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'border-prediction'
Bootstrap(app)

FILE_CONFIG = r'./config.json'
DAYS = 7

class DatePicker(Form): 
    date_selected = DateField(id='datepick')   
    def __init__(self, date_min, date_max, *args, **kwargs): 
        super(DatePicker, self).__init__(*args, **kwargs)
        self.date_min = date_min
        self.date_max = date_max        
    
def read_json_file(filename):
    with open(filename) as json_file:
        obj = json.load(json_file)
    return obj

# Load config variables    
config = read_json_file(FILE_CONFIG)

@app.route('/', methods=['GET','POST'])
def index(): 
    df_preds = pd.read_csv(config["file"]["FILE_PREDICTION"]) 
    df_preds['Date_time'] = pd.to_datetime(df_preds['Date_time'])
    df_preds['Date'] = df_preds['Date_time'].dt.date
    session["df_preds"] = df_preds.to_json()
    df_rmse = pd.read_csv(config["file"]["RMSE_RESULTS"])
    pred_img = './static/7_day_prediction.png'     
    date_max = df_preds['Date'].max()
    date_min = date_max - timedelta(days=DAYS-1)
    form = DatePicker(date_min,date_max)  
    form.date_selected.default = date_max
    form.date_selected.validators =[DateRange(min=date_min, max=date_max)]
    form.process()
    # Only need to display the last 30 days prediction RMSE
    return render_template('index.html', form=form, tables=[df_rmse[:30].to_html(classes='rmse',index=False,header=True)], pred_image = pred_img)

@app.route('/predictions')
def predictions():       
    df_preds=session.get('df_preds')
    df_preds=pd.read_json(df_preds, dtype=False)
    date_selected = request.args.get('jsdata')
    df_pred_selected = df_preds[df_preds['Date']==pd.Timestamp(date_selected)]
    return render_template('predictions.html',tables=[df_pred_selected[['Date_time','Expected Delay in Minutes']].to_html(classes='data',index=False,header=False)], titles=df_pred_selected.columns.values)


if __name__ == '__main__':    
    app.run(host="0.0.0.0", port=5000)
    