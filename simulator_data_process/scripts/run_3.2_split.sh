#!/bin/bash
for i in {0..15}; do
	echo $i
	nohup python run_3.2_split.py $i > ../res/log_3.2_${i}.log &
done
