# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2021/4/1
@function: 序列标注
"""
import os
import json
import time
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)


def add_arguments():
    parser=argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="duee-1-role", help="duee-1-trigger/duee-1-role/duee-fin-trigger/duee-fin-role")
    parser.add_argument("--name", type=str, default="hfl/chinese-roberta-wwm-ext", help="The pretrain model")
    parser.add_argument("--tag_path", type=str, default="./origin_data/duee_event_role_tag.json", help="")
    parser.add_argument("--train_path", type=str, default="./origin_data/role/train.tsv")
    parser.add_argument("--dev_path", type=str, default="./origin_data/role/dev.tsv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=int, default=0.0001)
    parser.add_argument("--max_length", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    args=parser.parse_args()
    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class InputFeature:
    def __init__(self, input_ids, input_mask, input_type_ids, labels=None, tokens=None):
        self.input_ids=input_ids
        self.input_mask=input_mask
        self.input_type_ids=input_type_ids
        self.labels=labels
        self.tokens=tokens


def from_file(data_path, tokenizer, label_map, max_length=512):
    features=[]
    with open(data_path, encoding="utf8") as fn:
        next(fn)
        lines=fn.readlines()[:10]
        for line in tqdm(lines):
            words, labels=line.strip().split("\t")
            words=words.split("\002")
            labels=labels.split("\002")
            feature=convert_example_to_feature((words, labels), tokenizer, label_map, max_length)
            features.append(feature)
    return features


def convert_example_to_feature(example, tokenizer, label_vocab=None, max_length=512):
    tokens, labels = example
    tokens=tokens[:max_length-2]
    labels=labels[:max_length-2]
    tokens=["CLS"]+tokens+["SEP"]
    labels=["O"]+labels+["O"]

    input_ids=tokenizer.convert_tokens_to_ids(tokens)
    label_ids=[label_vocab[label] for label in labels]
    input_mask=[1]*len(tokens)
    while len(input_ids)<max_length:
        input_ids.append(0)
        input_mask.append(0)
        label_ids.append(0)

    assert len(input_ids)==max_length
    assert len(input_mask)==max_length
    assert len(label_ids)==max_length

    input_type_ids=[0]*max_length
    feature=InputFeature(input_ids, input_mask, input_type_ids, labels=label_ids, tokens=tokens)
    return feature


def batch_generator(features, batch_size, shuffle=True):
    counts=len(features)
    idx_=range(counts)
    if shuffle:
        idx_=np.random.permutation(counts)
    batch=[]
    for i in idx_:
        batch.append(features[i])
        if len(batch)==batch_size or i==idx_[-1]:
            batch_input_ids=torch.tensor([feature.input_ids for feature in batch], dtype=torch.long)
            batch_input_mask=torch.tensor([feature.input_mask for feature in batch], dtype=torch.long)
            batch_input_type_ids=torch.tensor([feature.input_type_ids for feature in batch], dtype=torch.long)
            batch_labels=torch.tensor([feature.labels for feature in batch], dtype=torch.long)
            yield batch_input_ids, batch_input_mask, batch_input_type_ids, batch_labels
            batch=[]


def do_train(args):
    device="cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained(args.name)
    label_map = json.load(open(args.tag_path, encoding="utf8"))
    train_features=from_file(args.train_path, tokenizer, label_map, max_length=args.max_length)
    dev_features=from_file(args.dev_path, tokenizer, label_map, max_length=args.max_length)
    logger.info(f"Loading train dataset: {len(train_features)}, dev dataset: {len(dev_features)}")

    if len(train_features)%args.batch_size==0:
        one_epoch_steps=len(train_features)//args.batch_size
    else:
        one_epoch_steps = len(train_features) // args.batch_size + 1
    total_steps=one_epoch_steps*args.epochs
    logger.info(f"Total train steps: {total_steps}")

    bert=BertForTokenClassification.from_pretrained(args.name, num_labels=len(label_map)).to(device)
    optimizer=AdamW(bert.parameters(), lr=args.lr)
    scheduler=get_linear_schedule_with_warmup(optimizer, int(total_steps*0.1),total_steps)

    min_eval_loss = float("inf")
    for epoch in range(args.epochs):
        train_generator = batch_generator(train_features, args.batch_size)
        dev_generator = batch_generator(dev_features, args.batch_size, False)
        logger.info(f"======== Epoch {epoch + 1:} / {args.epochs:} ========")
        logger.info("Training...")
        bert.train()
        start_train = time.time()
        total_train_loss = 0
        for step, batch in enumerate(train_generator):
            batch_input_ids = batch[0].to(device=device)
            batch_input_mask = batch[1].to(device=device)
            batch_type_ids = batch[2].to(device=device)
            batch_labels = batch[3].to(device=device)
            outputs = bert(batch_input_ids, batch_input_mask, batch_type_ids, labels=batch_labels)
            bert.zero_grad()
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += outputs.loss.item()

            if step % 100 == 0:
                logger.info(f"  Step: {step + 1:>5}/{one_epoch_steps:>1}, current loss: {outputs.loss.item():.6f}")

        average_train_loss = total_train_loss / (step + 1)
        training_time = time.time() - start_train
        logger.info(f"  Average training loss: {average_train_loss:.6f}; Take time: {training_time:.3f}")

        logger.info("Running Validation...")
        bert.eval()
        start_eval = time.time()
        total_eval_loss = 0
        for step, batch in enumerate(dev_generator):
            batch_input_ids = batch[0].to(device=device)
            batch_input_mask = batch[1].to(device=device)
            batch_type_ids = batch[2].to(device=device)
            batch_labels = batch[3].to(device=device)
            with torch.no_grad():
                outputs = bert(batch_input_ids, batch_input_mask, batch_type_ids, labels=batch_labels)
                total_eval_loss += outputs.loss.item()
        average_eval_loss = total_eval_loss / (step + 1)
        validation_time = time.time() - start_eval
        logger.info(f"  Average eval loss: {average_eval_loss:.6f}; Take time: {validation_time:.3f}")

        if average_eval_loss<min_eval_loss:
            min_eval_loss=average_eval_loss
            logger.info("  Save model...")
            torch.save(bert.state_dict(),f"{args.task}_model_{epoch}.pt")







if __name__ == '__main__':
    setup_seed(1)
    args=add_arguments()
    do_train(args)
