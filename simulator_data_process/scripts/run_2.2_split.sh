#!/bin/bash
for i in 1 2 3 4 5; do
	echo $i
	nohup python run_2.2_split.py $i > ../res/log_2.2_${i}.log &
done
