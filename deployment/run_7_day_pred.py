# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 15:50:02 2019

@author: wangp
"""

from flask import Flask, request, render_template
import pandas as pd
from datetime import datetime, timedelta
from wtforms.fields.html5 import DateField
from wtforms_components import DateRange
from flask_wtf import Form
from flask_bootstrap import Bootstrap

app = Flask(__name__)
app.config['SECRET_KEY'] = 'border-prediction'
Bootstrap(app)
FILE_PREDICTION = r'./results/7_day_forecast.csv'
FILE_RMSE = r'./results/rmse_results.csv'

df_preds = pd.read_csv(FILE_PREDICTION) 
df_preds['Date_time'] = pd.to_datetime(df_preds['Date_time'])
df_preds['Date'] = df_preds['Date_time'].dt.date
df_rmse = pd.read_csv(FILE_RMSE) 
pred_img = './static/7_day_prediction.png'
DAYS = 7

class DatePicker(Form):
    date_max = df_preds['Date'].max()
    date_min = date_max - timedelta(days=DAYS-1)
    min_date = DateField(id='date_min_field',default=date_min)
    max_date = DateField(id='date_max_field',default=date_max)
    date_selected = DateField(id='datepick',default=date_max, validators=[DateRange(min=date_min, max=date_max)])
        
@app.route('/', methods=['GET','POST'])
def index(): 
    form = DatePicker(id="myform")        
    # Only need to display the last 30 days prediction RMSE
    return render_template('index.html', form=form, tables=[df_rmse[:30].to_html(classes='rmse',index=False,header=True)], pred_image = pred_img)

@app.route('/predictions')
def predictions():       
    date_selected = request.args.get('jsdata')
    df_pred_selected = df_preds[df_preds['Date']==datetime.strptime(date_selected,'%Y-%m-%d').date()]
    return render_template('predictions.html',tables=[df_pred_selected[['Date_time','Expected Delay in Minutes']].to_html(classes='data',index=False,header=False)], titles=df_pred_selected.columns.values)


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host="0.0.0.0", port=5000)
    