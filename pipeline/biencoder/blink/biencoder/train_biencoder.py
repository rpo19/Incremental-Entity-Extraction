# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import argparse
import pickle
import torch
import json
import sys
import io
import random
import time
import numpy as np

from multiprocessing.pool import ThreadPool

from tqdm import tqdm, trange
from collections import OrderedDict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.optimization import WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.modeling_utils import WEIGHTS_NAME

from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder
import logging

import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process as data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser

import base64
import requests


logger = None

# The evaluate function during training uses in-batch negatives:
# for a batch of size B, the labels from the batch are used as label candidates
# B is controlled by the parameter eval_batch_size
def evaluate(
    reranker, eval_dataloader, params, device, logger,
):
    reranker.model.eval()
    if params["silent"]:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input, candidate_input, _ = batch

        with torch.no_grad():
            eval_loss, logits = reranker(context_input, candidate_input)

        logits = logits.detach().cpu().numpy()
        # Using in-batch negatives, the label ids are diagonal
        label_ids = torch.LongTensor(
                torch.arange(params["eval_batch_size"])
        ).numpy()
        tmp_eval_accuracy, _ = utils.accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += context_input.size(0)

        nb_eval_steps += 1

    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
    results["normalized_accuracy"] = normalized_eval_accuracy
    return results


def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
    )


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
    )
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler

def vector_encode(v):
    s = base64.b64encode(v).decode()
    return s

def main(params):
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model

    device = reranker.device
    n_gpu = reranker.n_gpu

    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # Load train data
    # TODO get 9M randomly at every epoch?
    train_samples = utils.read_dataset("train", params["data_path"], compression='gzip',
        max=params['max_dataset'], sample=params['sample_dataset'], seed=params['sample_dataset_seed'])
    logger.info("Read %d train samples." % len(train_samples))

    train_dataloader = data.process_mention_data_iter(
        train_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        context_key=params["context_key"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
        batch_size=train_batch_size,
        label_key="descr",
        title_key='href',
        label_idx_key='target',
        start_from_instance=params["start_from_instance"]
    )

    # Load eval data
    # TODO: reduce duplicated code here
    valid_samples = utils.read_dataset("valid", params["data_path"], compression='gzip')
    logger.info("Read %d valid samples." % len(valid_samples))

    valid_dataloader = data.process_mention_data_iter(
        valid_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        context_key=params["context_key"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
        batch_size=eval_batch_size,
        label_key="descr",
        title_key='href',
        label_idx_key='label',
    )

    # evaluate before training
    results = evaluate(
        reranker, valid_dataloader, params, device=device, logger=logger,
    )

    valid_dataloader = data.process_mention_data_iter(
        valid_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        context_key=params["context_key"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
        batch_size=eval_batch_size,
        label_key="descr",
        title_key='href',
        label_idx_key='label',
    )

    number_of_samples_per_dataset = {}

    time_start = time.time()

    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )

    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )

    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(params, optimizer, len(train_samples), logger)

    model.train()

    best_epoch_idx = -1
    best_score = -1

    num_train_epochs = params["num_train_epochs"]
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        results = None

        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")

        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            context_input, candidate_input, label_input = batch

            # if params['hard_negatives']:
            #     with torch.no_grad():
            #         context_input = context_input.to(reranker.device)
            #         context_encoding = reranker.encode_context(context_input).numpy()
            #         context_encoding = np.ascontiguousarray(context_encoding)
            #     encodings = [vector_encode(e) for e in encodings]

            #     body = {
            #         'encodings': encodings,
            #         'top_k': 2 # to ensure there is also a negative
            #     }
            #     res_indexer = requests.post(params['indexer_url'], json=body)
            #     # TODO do not risk to compromise the entire training for a single failure here
            #     assert res_indexer.ok
            #     candidates = res_indexer.json()

            #     # TODO remove: dataset with hard negatives is created in advance

            # if hard negs pass label_input
            if params['hard_negatives']:
                # n -> n,1
                label_input = torch.reshape(label_input, (label_input.shape[0],))
                # label_input = label_input.type(torch.float)
                # label_input.to(reranker.device)
                context_input = context_input[label_input == 1]
                loss, _ = reranker(context_input, candidate_input, random_negs=False)
            else:
                loss, _ = reranker(context_input, candidate_input)

            # if n_gpu > 1:
            #     loss = loss.mean() # mean() to average on multi-gpu.

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                logger.info(
                    "Step {} - epoch {} average loss: {}\n".format(
                        step,
                        epoch_idx,
                        tr_loss / (params["print_interval"] * grad_acc_steps),
                    )
                )
                tr_loss = 0

            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                logger.info("Evaluation on the development dataset")
                results = evaluate(
                    reranker, valid_dataloader, params, device=device, logger=logger,
                )
                logger.info("***** Saving fine - tuned model *****")
                epoch_output_folder_path = os.path.join(
                    model_output_path, "epoch_{}_{}".format(epoch_idx, step)
                )
                utils.save_model(model, tokenizer, epoch_output_folder_path)

                output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
                with open(output_eval_file, 'w') as fd:
                    json.dump(results, fd)
                # reset dataloader TODO improve
                valid_dataloader = data.process_mention_data_iter(
                    valid_samples,
                    tokenizer,
                    params["max_context_length"],
                    params["max_cand_length"],
                    context_key=params["context_key"],
                    silent=params["silent"],
                    logger=logger,
                    debug=params["debug"],
                    batch_size=eval_batch_size,
                    label_key="descr",
                    title_key='href',
                    label_idx_key='label',
                )
                model.train()
                logger.info("\n")

        train_dataloader = data.process_mention_data_iter(
            train_samples,
            tokenizer,
            params["max_context_length"],
            params["max_cand_length"],
            context_key=params["context_key"],
            silent=params["silent"],
            logger=logger,
            debug=params["debug"],
            batch_size=train_batch_size,
            label_key="descr",
            title_key='href',
            label_idx_key='target',
        )

        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(model, tokenizer, epoch_output_folder_path)

        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        results = evaluate(
            reranker, valid_dataloader, params, device=device, logger=logger,
        )
        with open(output_eval_file, 'w') as fd:
            json.dump(results, fd)
        # reset dataloader TODO improve
        valid_dataloader = data.process_mention_data_iter(
            valid_samples,
            tokenizer,
            params["max_context_length"],
            params["max_cand_length"],
            context_key=params["context_key"],
            silent=params["silent"],
            logger=logger,
            debug=params["debug"],
            batch_size=eval_batch_size,
            label_key="descr",
            title_key='href',
            label_idx_key='label',
        )

        ls = [best_score, results["normalized_accuracy"]]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        logger.info("\n")

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best model in the parent_dir
    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    params["path_to_model"] = os.path.join(
        model_output_path,
        "epoch_{}".format(best_epoch_idx),
        WEIGHTS_NAME,
    )
    reranker = load_biencoder(params)
    utils.save_model(reranker.model, tokenizer, model_output_path)

    if params["evaluate"]:
        params["path_to_model"] = model_output_path
        evaluate(params, logger=logger)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()
    parser.add_argument(
        "--max-dataset", default=None, type=int, dest='max_dataset',
        help="Limit the dataset to this size."
    )
    parser.add_argument(
        "--sample-dataset", default=None, type=int, dest='sample_dataset',
        help="Sample the dataset to this size."
    )
    parser.add_argument(
        "--sample-dataset-seed", default=None, type=int, dest='sample_dataset_seed',
        help="Sample with this seed."
    )
    parser.add_argument(
        #TODO int for how many hard negatives?
        "--hard-negatives", action="store_true", help="Whether to use hard-negatives.",
        dest='hard_negatives', default=False
    )
    parser.add_argument(
        "--indexer-url",
        default=None,
        type=str,
        help="The url of the indexer from where to extract hard negatives.",
        dest='indexer_url'
    )

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
