#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/anaconda3/lib/ >> /home/auto_disc/out 2>> /home/auto_disc/error;

PIDS=""
for ((i=0; i<$1; i++))
do
   PID=$(/root/anaconda3/envs/autoDiscTool/bin/python3 /home/auto_disc/auto_disc/auto_disc/run.py --seed ${i} ${@:2} >> experiment_logs.log & echo $!)
   if [[ $PIDS != "" ]]
   then
       PIDS+=" "
   fi
   PIDS+="${PID}"
done
echo "[RUN_ID_start]$PIDS[RUN_ID_stop]"