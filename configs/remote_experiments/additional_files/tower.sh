#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mayalen/anaconda3/lib/;

PIDS=""
for ((i=0; i<$1; i++))
do
   PID=$(/home/mayalen/anaconda3/envs/autoDiscTool/bin/python3 /home/mayalen/code/09-AutoDisc/experiments/libs/auto_disc/run.py --seed ${i} ${@:2} >> experiment_logs.log & echo $!)
   if [[ $PIDS != "" ]]
   then
       PIDS+=" "
   fi
   PIDS+="${PID}"
done
echo "[RUN_ID_start]$PIDS[RUN_ID_stop]"