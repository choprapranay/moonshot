#!/bin/bash

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

pip install -r requirements.txt

echo "source venv/bin/activate && python app.py"
