#!/bin/bash
for i in 2 4 6 8 10 12 15; do
	echo $i
	nohup python run_1.2_split.py $i > ../res/log_1.2_${i}.log &
done
