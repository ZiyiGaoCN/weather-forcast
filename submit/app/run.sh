#!/bin/bash

# 这里可以放入代码运行命令
echo "program start..."
export LD_LIBRARY_PATH=/usr/local/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
python3 run.py