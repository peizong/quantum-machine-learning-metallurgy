#!/bin/bash
for i in 100 300 500 700 900 1100 1250 1252; do
	echo $i
	nohup python run_5_split.py $i > ../res/log_5_${i}.log &
done
