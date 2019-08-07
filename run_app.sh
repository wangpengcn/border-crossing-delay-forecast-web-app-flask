#!/bin/bash
#Run Flask app

FOLDER_PATH="/home/ec2-user/border_forecast/"
VENV_NAME="venv_border_forecast"

#run flask
export FLASK_APP=$FOLDER_PATH/deployment/run_7_day_pred.py
$VENV_NAME/bin/python3 -m flask run --host=0.0.0.0