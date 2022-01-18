#!/bin/bash

PIDS=""
for ((i=1; i<=$1; i++))
do
   PID=$(python run.py --seed ${i} ${@:2} >> experiment_logs.log & echo $!)
   if [[ $PIDS != "" ]]
   then
       PIDS+=" "
   fi
   PIDS+="${PID}"
done
echo $PIDS
