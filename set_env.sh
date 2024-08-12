#!/bin/bash
pip install -r requirements.txt

export DATA_DIR="$(pwd)/Pascal-part"
PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONPATH