#!/bin/bash
# UVR5 快速處理工具 - Shell 腳本
# 使用方式: ./uvr5.sh input.wav 或 ./uvr5.sh "back*.wav"

python src/uvr5_cli.py "$@"