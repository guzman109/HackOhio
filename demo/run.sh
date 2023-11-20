#!/bin/bash
#
export QT_QPA_PLATFORM=xcb;
export PYOPENGL_PLATFORM=glx;
export MODEL_REPO=~/Documents/HackOhio/model_repository/
~/Documents/HackOhio/demo-local-venv/bin/python3.9 ~/Documents/HackOhio/demo/stream-local.py 
