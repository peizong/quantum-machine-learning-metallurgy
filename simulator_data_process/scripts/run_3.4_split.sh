#!/bin/bash
for i in {0..15}; do
	echo $i
	nohup python run_3.4_split.py $i > ../res/log_3.4_${i}.log &
done
