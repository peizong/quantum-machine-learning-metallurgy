#!/bin/bash
for i in {0..15}; do
	echo $i
	nohup python run_3.3_split.py $i > ../res/log_3.3_${i}.log &
done
