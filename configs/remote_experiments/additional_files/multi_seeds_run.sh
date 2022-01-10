#!/bin/bash

PIDS=""
for ((i=1; i<=$1; i++))
do
   PID=$(python test.py ${@:2} >> experiment_logs.log & echo $!)
   if [[ $PIDS != "" ]]
   then
       PIDS+=" "
   fi
   PIDS+="${PID}"
done
echo $PIDS
