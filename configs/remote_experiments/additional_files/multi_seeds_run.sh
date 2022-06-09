#!/bin/bash

PIDS=""
for ((i=1; i<=$1; i++))
do
   PID=$(python auto_disc/auto_disc/run.py --seed ${i} ${@:2} >> experiment_logs.log & echo $!)
   if [[ $PIDS != "" ]]
   then
       PIDS+=" "
   fi
   PIDS+="${PID}"
done
echo "[RUN_ID_start]$PIDS[RUN_ID_stop]"
