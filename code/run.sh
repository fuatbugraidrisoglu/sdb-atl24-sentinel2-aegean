#!/usr/bin/env bash
# Code Ocean entry point. Reproduces all reported quantitative results.
set -e
cd "$(dirname "$0")"
python -u run.py
