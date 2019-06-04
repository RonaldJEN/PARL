#!/bin/bash


export CPU_NUM=1

for i in $(seq 1 32); do
    nohup /home/zhoubo01/tools/miniconda2/envs/es/bin/python actor.py &
done;
wait
