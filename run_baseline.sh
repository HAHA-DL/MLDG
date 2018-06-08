#!/usr/bin/env bash

times=5
for j in `seq 1 $times`
do
max=3
for i in `seq 0 $max`
do
python main_baseline.py train \
--lr=5e-4 \
--num_classes=7 \
--test_every=500 \
--logs='run_'$j'/logs_'$i'/' \
--batch_size=64 \
--model_path='run_'$j'/models_'$i'/' \
--unseen_index=$i \
--inner_loops=45001 \
--step_size=15000 \
--state_dict='' \
--data_root=$1
done
done