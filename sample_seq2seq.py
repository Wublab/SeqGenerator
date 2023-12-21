"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json
import torch
from tracemalloc import start

import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from diffuseq.rounding import denoised_fn_round, get_weights
from diffuseq.text_datasets import load_data_text
import pandas as pd
import random

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import time
from diffuseq.utils import dist_util, logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_model_emb,
    load_tokenizer
)

def create_argparser():
    defaults = dict(model_path='', model_diff_path='', model_regression_path = '',step=0, out_dir='', top_p=0, \
                    seq_len_sample='', max_len=0, min_len=0, seq_num=0)
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)  
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

token2ids = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}
ids2token = {}
for key in token2ids:
    ids2token[token2ids[key]] = key

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    # load configurations.
    config_path = os.path.join(os.path.split(args.model_diff_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)


    logger.log("### Creating model and diffusion...")
    model, diffusion, alphabet, model_state = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_diff_path, map_location="cpu")
    )
    #model.load_state_dict(model_state)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.to(dist_util.dev())
    model.eval()

    #tokenizer = load_tokenizer(args)
    model_emb = load_model_emb(args, model.embed_tokens.weight)

    model_emb.weight = th.nn.Parameter(model.embed_tokens.weight.clone().cpu())
    model_emb_copy = get_weights(model_emb, args)

    set_seed(args.seed2)

    print("### Sampling...on", args.split)


    start_t = time.time()
    
    # batch, cond = next(data_valid)
    # print(batch.shape)

    model_base_name = os.path.basename(os.path.split(args.model_diff_path)[0]) + f'.{os.path.split(args.model_diff_path)[1]}'
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}.json")
    # fout = open(out_path, 'a')

    
    from tqdm import tqdm


    pdb_csv = pd.read_csv(args.seq_len_sample)
    pdb_csv = pdb_csv[pdb_csv.len <= args.max_len]
    pdb_csv = pdb_csv[pdb_csv.len >= args.min_len]

    data_list = pdb_csv.len.values.tolist()

    random.shuffle(data_list)
    seq_list = data_list[:args.seq_num]

    data_dict = {}
    for item in seq_list:
        if item not in data_dict:
            data_dict[item] = 0
        data_dict[item] += 1

    for seq_len in tqdm(range(args.max_len, args.min_len - 1, -1)):
        if seq_len not in data_dict:
            continue
        for i in range(int(data_dict[seq_len]/4) + 1):
            if (i + 1)*4 <= data_dict[seq_len]:
                count = 4
            else:
                count = data_dict[seq_len] - i*4
            if count == 0:
                continue
            used_len = seq_len + 2

        
            input_ids_x = torch.tensor([[5]*used_len]*count).to(dist_util.dev())

            input_ids_mask = torch.ones((count, used_len), dtype=torch.int64)

            input_ids_mask_ori = input_ids_mask

            noise = th.randn(count, used_len, 640)

            input_ids_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), noise.shape).to(dist_util.dev())
            x_noised = noise.to(dist_util.dev())

            model_kwargs = {}

            if args.step == args.diffusion_steps:
                args.use_ddim = False
                step_gap = 1
            else:
                args.use_ddim = True
                step_gap = args.diffusion_steps//args.step


            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )

            sample_shape = (x_noised.shape[0], seq_len, x_noised.shape[-1])

            samples = sample_fn(
                model,
                sample_shape,
                noise=x_noised,
                clip_denoised=args.clip_denoised,
                denoised_fn=partial(denoised_fn_round, args, model_emb_copy.cuda()),
                model_kwargs=model_kwargs,
                top_p=args.top_p,
                clamp_step=args.clamp_step,
                clamp_first=True,
                mask=input_ids_mask,
                x_start=input_ids_x,
                gap=step_gap
            )

            model_emb_copy.cpu()


            sample = samples[-1]
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)
            all_sentence = [sample.cpu().numpy() for sample in gathered_samples]

            # print('sampling takes {:.2f}s .....'.format(time.time() - start_t))

            word_lst_recover = []
            word_lst_ref = []
            word_lst_source = []


            arr = np.concatenate(all_sentence, axis=0)
            x_t = th.tensor(arr).cuda()


            reshaped_x_t = x_t
            logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab

            cands = th.topk(logits, k=1, dim=-1)
            sample = cands.indices
            # tokenizer = load_tokenizer(args)

            for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
                tokens = decode_seq(seq)
                word_lst_recover.append(tokens)


            fout = open(out_path, 'a')
            for recov in word_lst_recover:
                print(json.dumps({"len": seq_len, "recover": recov}), file=fout)
            fout.close()

    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')

def decode_seq(seq):
    seq_str = ''
    for item in seq.squeeze().tolist():
        seq_str += ids2token[item]
    return seq_str


if __name__ == "__main__":
    main()
