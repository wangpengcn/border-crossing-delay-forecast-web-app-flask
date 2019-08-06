#!/bin/bash
#1. git clone https://github.com/wangpengcn/border-crossing-delay-forecast-web-app-flask.git border_forecast
#2. cd border_forecast
#3. chmod 700 run.sh (read, write, and execute (run) the script -- but only your user)
#4. ./run.sh

FOLDER_PATH="/home/ec2-user/border_forecast/"
FOLDER_NAME="border_forecast"
VENV_NAME="venv_border_forecast"

#Perform a yum install update
sudo yum update

#List of all python3 packages available
#sudo yum list | grep python3
#Install Python3.6
sudo yum install python36-pip

# Install virtualenv using pip3
pip3 install --user virtualenv

# Create a virtual environment
python3 -m venv $VENV_NAME

# Activate virtual environment
source $VENV_NAME/bin/activate

#Install required python packages
$VENV_NAME/bin/pip3 install -r requirements.txt

# schedule 05:00 everyday to rebuild prediction model and generate forecast for next 7 days
(crontab -l 2>/dev/null; echo "0 5 * * * cd $FOLDER_PATH && $VENV_NAME/bin/python3 border_wait_time_forecast.py > /tmp/border.log") | crontab -

#run flask
export FLASK_APP=./deployment/run_7_day_pred.py
$VENV_NAME/bin/python3 -m flask run --host=0.0.0.0