#!/bin/bash
for i in 100 300 500 700 900 1100 1250 1252; do
	echo $i
	nohup python run_4.1_split.py $i > ../res/log_4.1_${i}.log &
done
