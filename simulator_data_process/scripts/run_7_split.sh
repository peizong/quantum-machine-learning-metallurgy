#!/bin/bash
for i in 0 1 2 3 4 5; do
	echo $i
	nohup python run_7_split.py $i > ../res/log_7_${i}.log &
done
