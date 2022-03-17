# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import os
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
import multiprocessing
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data
from configs import set_seed, set_dist


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    filename='results/bleu.log',
                    filemode='a',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            preds = model.generate(source_ids,
                                   attention_mask=source_mask,
                                   use_cache=True,
                                   num_beams=args.beam_size,
                                   early_stopping=args.task == 'summarize',
                                   max_length=args.max_target_length)
            top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]

    output_fn = os.path.join(args.res_dir, "{}_test_{}.output".format(args.dataset, criteria))
    gold_fn = os.path.join(args.res_dir, "{}_test_{}.gold".format(args.dataset, criteria))
    src_fn = os.path.join(args.res_dir, "{}_test_{}.src".format(args.dataset, criteria))

    if args.task in ['defect']:
        target_dict = {0: 'false', 1: 'true'}
        golds = [target_dict[ex.target] for ex in eval_examples]
        eval_acc = np.mean([int(p == g) for p, g in zip(pred_nls, golds)])
        result = {'em': eval_acc * 100, 'bleu': 0, 'codebleu': 0}

        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                f.write(pred_nl.strip() + '\n')
                f1.write(target_dict[gold.target] + '\n')
                f2.write(gold.source.strip() + '\n')
            logger.info("Save the predictions into %s", output_fn)
    else:
        dev_accs, predictions = [], []
        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                dev_accs.append(pred_nl.strip() == gold.target.strip())
                if args.task in ['summarize']:
                    # for smooth-bleu4 evaluation
                    predictions.append(str(gold.idx) + '\t' + pred_nl)
                    f.write(str(gold.idx) + '\t' + pred_nl.strip() + '\n')
                    f1.write(str(gold.idx) + '\t' + gold.target.strip() + '\n')
                    f2.write(str(gold.idx) + '\t' + gold.source.strip() + '\n')
                else:
                    f.write(pred_nl.strip() + '\n')
                    f1.write(gold.target.strip() + '\n')
                    f2.write(gold.source.strip() + '\n')

        if args.task == 'summarize':
            (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
            bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        else:
            bleu = round(_bleu(gold_fn, output_fn), 2)
            if args.task in ['concode', 'translate', 'refine']:
                codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, args.lang)

        result = {'em': np.mean(dev_accs) * 100, 'bleu': bleu}
        if args.task == 'concode':
            result['codebleu'] = codebleu * 100

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def add_args(parser):
    parser.add_argument("--model_tag", type=str, default='codet5_base',
                        choices=['roberta', 'codebert', 'bart_base', 'codet5_small', 'codet5_base'])
    parser.add_argument("--task", type=str, default='summarize', choices=['summarize', 'concode', 'translate',
                                                                          'refine', 'defect', 'clone', 'multi_task'])
    parser.add_argument("--sub_task", type=str, default='pytorch')
    parser.add_argument("--res_dir", type=str, default='results', help='directory to save fine-tuning results')
    parser.add_argument("--model_dir", type=str, default='saved_models', help='directory to save fine-tuned models')
    parser.add_argument("--summary_dir", type=str, default='tensorboard', help='directory to save tensorboard summary')
    parser.add_argument("--cache_path", type=str, default='cache')
    parser.add_argument("--data_num", type=int, default=-1, help='number of data instances to use, -1 for full data')
    parser.add_argument("--gpu", type=int, default=0, help='index of the gpu to use in a cluster')
    parser.add_argument("--lang", type=str, default='python')
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument("--add_task_prefix", action='store_true', help="Whether to add task prefix for t5 and codet5")
    parser.add_argument("--res_fn", type=str, default='results/summarize_codet5.txt')

    parser.add_argument("--model_type", default="codet5", type=str, choices=['roberta', 'bart', 'codet5'])
    parser.add_argument("--add_lang_ids", action='store_true')

    ## Required parameters
    parser.add_argument("--model_name_or_path", default="pretrained_models/codet5_base", type=str,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default='results', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default='saved_models/summarize/pytorch/codet5_base_all_lr5_bs48_src256_trg128_pat2_e15/checkpoint-best-bleu/pytorch_model.bin', type=str,
                        help="Path to trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default='data/dataset/output.jsonl', type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dataset", default='pytorch', type=str,
                        help="cache file name")

    parser.add_argument("--config_name", default="Salesforce/codet5-base", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="Salesforce/codet5-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=512, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=256, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=True, action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")
    args = parser.parse_args()

    return args


def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)

    pool = multiprocessing.Pool(args.cpu_cont)
    if args.test_filename is None or args.test_filename.strip() == '':
        _, _, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)

    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_examples, eval_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, f'{args.dataset}_test',
                                                           only_src=True, is_sample=False)
        result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'test', 'best-bleu')
        test_bleu, test_em = result['bleu'], result['em']
        test_codebleu = result['codebleu'] if 'codebleu' in result else 0
        result_str = "[%s] bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % ('best-bleu', test_bleu, test_em, test_codebleu)
        logger.info(result_str)
        print(result_str)
        fa.write(result_str)
        if args.res_fn:
            with open(args.res_fn, 'a+') as f:
                f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), args.model_name_or_path))
                f.write(result_str)
    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


if __name__ == "__main__":
    main()
