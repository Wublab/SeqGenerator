#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=1 --master_port=12236 --use_env run_train.py \
--diff_steps 2000 \
--lr 0.00005 \
--learning_steps 1000000 \
--save_interval 10000 \
--seed 102 \
--noise_schedule sqrt \
--bsz 4 \
--max_len 490 \
--min_len 460 \
--dataset qqp \
--data_dir ./datasets/aspartese \
--schedule_sampler lossaware \
--notes test-qqp \
--model_path ./diffusion_models/esm_orig/esm2_t30_150M_UR50D.pt \
--model_regression_path ./diffusion_models/esm_orig/esm2_t30_150M_UR50D-contact-regression.pt
