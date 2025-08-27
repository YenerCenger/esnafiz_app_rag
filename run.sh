#!/usr/bin/env bash
cd "$(dirname "$0")"
export PYTHONPATH=$(pwd)
uvicorn main:app --host 0.0.0.0 --port 8000
