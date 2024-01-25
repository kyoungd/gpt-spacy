#!/bin/bash
source ./venv/bin/activate

export FLASK_APP=app.py
export FLASK_ENV=development  # optional, only if you want development mode
flask run --port=3011  # or any other port you want
