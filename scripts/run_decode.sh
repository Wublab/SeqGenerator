#!/bin/bash

python -u run_decode.py \
--model_dir diffusion_models/select \
--seq_len_sample datasets/aspartese/train.csv \
--max_len 490 \
--min_len 460 \
--seq_num 500 \
--seed 123 \
--split test
