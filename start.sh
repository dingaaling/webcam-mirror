#!/bin/bash

# run the app
python main_fb.py --ip 0.0.0.0 --port 8000 &
python main_gov.py --ip 0.0.0.0 --port 8080 &


# open in new browser tab
# xdg-open for linux | start for windows
# python -m webbrowser

open http://0.0.0.0:8000/ &
open http://0.0.0.0:8080/
