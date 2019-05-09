#!/bin/bash

#Start surgery notification daemon
nohup python scheduled_tasks.py > /dev/null 2>&1 &

#Start flask server
export FLASK_APP=app/__init__.py
export FLASK_DEBUG=1
flask run --host=0.0.0.0 --without-threads
