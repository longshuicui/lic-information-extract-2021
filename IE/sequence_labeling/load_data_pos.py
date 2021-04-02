# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2021/3/20
@function:
"""
import json
import logging
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, SequentialSampler, RandomSampler, DataLoader
from transformers import BertTokenizer

from IE.sequence_labeling.extract_chinese_and_punts import ChineseAndPunctuationExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFeature:
    def __init__(self, input_ids, input_mask, labels=None, tokens=None, token_to_origin_start_index=None,
                 token_to_origin_end_index=None, raw_text=None):
        assert len(input_ids) == len(labels)
        assert len(input_ids) == len(input_mask)
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = [0] * len(input_ids)
        self.labels = labels
        self.tokens = tokens
        self.token_to_origin_start_index = token_to_origin_start_index
        self.token_to_origin_end_index = token_to_origin_end_index
        self.raw_text = raw_text


def parse_label(spo_list, label_map, tokens, tokenizer):
    seq_len = len(tokens)
    num_labels = len(label_map)
    labels = [[0] * num_labels for _ in range(seq_len)]

    if spo_list is not None:
        for spo in spo_list:
            sub = spo["subject"]
            obj_type = spo["object_type"]
            predicate = spo["predicate"]
            for spo_object in spo["object"]:
                try:
                    label = predicate + "-" + obj_type[spo_object]
                except KeyError:
                    # print(spo_object)
                    label = predicate + "-" + "Number"
                obj = spo["object"][spo_object]

                subject_label_id = label_map["S-" + label]  # 主体的label id
                object_label_id = label_map["O-" + label]  # 客体的label id
                subject_tokens = tokenizer.tokenize(sub)
                object_tokens = tokenizer.tokenize(obj)

                subject_tokens_len = len(subject_tokens)
                object_tokens_len = len(object_tokens)

                # 分配token的label，这里根据不同的标注方式处理方式不同
                forbidden_index = None
                if subject_tokens_len > object_tokens_len:
                    for index in range(seq_len - subject_tokens_len + 1):
                        if tokens[index:index + subject_tokens_len] == subject_tokens:
                            labels[index][subject_label_id] = 1
                            for i in range(1, subject_tokens_len):
                                labels[index + i][1] = 1
                            forbidden_index = index
                            break
                    for index in range(seq_len - object_tokens_len + 1):
                        if tokens[index:index + object_tokens_len] == object_tokens:
                            if forbidden_index is None:
                                labels[index][object_label_id] = 1
                                for i in range(1, object_tokens_len):
                                    labels[index + i][1] = 1
                                break
                            elif index < forbidden_index or index >= forbidden_index + subject_tokens_len:
                                labels[index][object_label_id] = 1
                                for i in range(1, object_tokens_len):
                                    labels[index + i][1] = 1
                                break
                else:
                    for index in range(seq_len - object_tokens_len + 1):
                        if tokens[index:index + object_tokens_len] == object_tokens:
                            labels[index][object_label_id] = 1
                            for i in range(1, object_tokens_len):
                                labels[index + i][1] = 1
                            forbidden_index = index
                            break
                    for index in range(seq_len - subject_tokens_len + 1):
                        if tokens[index:index + subject_tokens_len] == subject_tokens:
                            if forbidden_index is None:
                                labels[index][subject_label_id] = 1
                                for i in range(1, subject_tokens_len):
                                    labels[index + i][1] = 1
                                break
                            elif index < forbidden_index or index >= forbidden_index + object_tokens_len:
                                labels[index][subject_label_id] = 1
                                for i in range(1, subject_tokens_len):
                                    labels[index + i][1] = 1
                                break

    for i in range(seq_len):
        if labels[i] == [0] * num_labels:
            labels[i][0] = 1
    return labels


def convert_example_to_feature(example, tokenizer, label_map, chinese_and_punct_extractor, max_length=512):
    spo_list = example["spo_list"] if "spo_list" in example else None
    raw_text = example["text"]
    # 清洗数据，这里判断是不是中文或者标点符号
    sub_text = []
    buffer = ""
    for char in raw_text:
        if chinese_and_punct_extractor.is_chinese_or_punct(char):
            if buffer != "":
                sub_text.append(buffer)
                buffer = ""
            sub_text.append(char)
        else:
            buffer += char
    if buffer != "":
        sub_text.append(buffer)

    # 分词，记录索引，方便label分配
    token_to_origin_start_index = []  # 该token在源字符串中的起始索引
    token_to_origin_end_index = []  # 该token在源字符串中的结束索引
    origin_to_token_index = []  # 该字符在tokens集合里的某一个token的索引
    tokens = []
    text_tmp = ""

    for i, token in enumerate(sub_text):
        origin_to_token_index.append(len(tokens))
        sub_tokens = tokenizer.tokenize(token)
        text_tmp += token
        for sub_token in sub_tokens:
            token_to_origin_start_index.append(len(text_tmp) - len(token))
            token_to_origin_end_index.append(len(text_tmp) - 1)
            tokens.append(sub_token)
            if len(tokens) >= max_length - 2:
                break
        else:
            continue
        break

    # 解析label
    labels = parse_label(spo_list, label_map, tokens, tokenizer)

    if len(tokens) > max_length - 2:
        tokens = tokens[0:(max_length - 2)]
        labels = labels[0:(max_length - 2)]
        token_to_origin_start_index = token_to_origin_start_index[0:(max_length - 2)]
        token_to_origin_end_index = token_to_origin_end_index[0:(max_length - 2)]

    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    outside_label = [[1] + [0] * (len(label_map) - 1)]  # [CLS]和[SEP]的label都是O
    labels = outside_label + labels + outside_label
    token_to_origin_start_index = [-1] + token_to_origin_start_index + [-1]
    token_to_origin_end_index = [-1] + token_to_origin_end_index + [-1]
    input_mask = [1] * len(tokens)

    if len(tokens) < max_length:
        tokens = tokens + ["[PAD]"] * (max_length - len(tokens))
        labels = labels + outside_label * (max_length - len(labels))
        token_to_origin_start_index = token_to_origin_start_index + [-1] * (
                    max_length - len(token_to_origin_start_index))
        token_to_origin_end_index = token_to_origin_end_index + [-1] * (max_length - len(token_to_origin_end_index))
        input_mask = input_mask + [0] * (max_length - len(input_mask))

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    feature = InputFeature(input_ids=token_ids,
                           input_mask=input_mask,
                           labels=labels,
                           tokens=tokens,
                           token_to_origin_start_index=token_to_origin_start_index,
                           token_to_origin_end_index=token_to_origin_end_index,
                           raw_text=raw_text)
    return feature


def from_file(file_path, tokenizer, label_map, chinese_and_punct_extractor, max_length=320):
    logger.info(f"Preprocessing data, loaded from {file_path}.")
    features = []
    with open(file_path, encoding="utf8") as fp:
        lines = fp.readlines()[:10]
        for line in tqdm(lines):
            example = json.loads(line.strip())
            feature = convert_example_to_feature(example, tokenizer, label_map, chinese_and_punct_extractor, max_length)
            features.append(feature)
    return features


def batch_generator(features, batch_size, shuffle=True):
    np.random.seed(1)
    counts = len(features)
    permute_idx = range(counts)
    if shuffle:
        permute_idx = np.random.permutation(counts)
    batch = []
    for i, index in enumerate(permute_idx):
        batch.append(features[index])
        if len(batch) == batch_size or i == counts - 1:
            batch_input_ids = torch.tensor([feature.input_ids for feature in batch], dtype=torch.long)
            batch_input_mask = torch.tensor([feature.input_mask for feature in batch], dtype=torch.long)
            batch_input_type_ids = torch.tensor([feature.input_type_ids for feature in batch], dtype=torch.long)
            batch_labels = torch.tensor([feature.labels for feature in batch], dtype=torch.long)
            yield batch_input_ids, batch_input_mask, batch_input_type_ids, batch_labels
            batch = []


def batch_loader(features, batch_size, training=True):
    batch_input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)
    batch_input_mask = torch.tensor([feature.input_mask for feature in features], dtype=torch.long)
    batch_input_type_ids = torch.tensor([feature.input_type_ids for feature in features], dtype=torch.long)
    batch_labels = torch.tensor([feature.labels for feature in features], dtype=torch.long)
    dataset = TensorDataset(batch_input_ids, batch_input_mask, batch_input_type_ids, batch_labels)
    if training:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=RandomSampler(dataset))
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=SequentialSampler(dataset))
    return dataloader


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    cpe = ChineseAndPunctuationExtractor()
    label_map, id2label = json.load(open("../origin_data/labels2idx.json", encoding="utf8"))

    example = {
        "text": "泳儿首张精选专辑《Vin'Selection1》,以甜酸苦辣严选她出道五年以来的26首精选歌曲,包括∶出道代表作“感应”、“花无雪”,以及“黛玉笑了”、“无心恋唱”、“送我一个家”、“最美丽的第七天”、“小蛮腰”、“一撇”和与海鸣威合唱金曲“我的回忆不是我的”",
        "spo_list": [
            {
                "predicate": "所属专辑",
                "object_type": {"@value": "音乐专辑"},
                "subject_type": "歌曲",
                "object": {"@value": "vin'selection"},
                "subject": "无心恋唱"
            },
            {
                "predicate": "所属专辑",
                "object_type": {"@value": "音乐专辑"},
                "subject_type": "歌曲",
                "object": {"@value": "Vin'Selection1"},
                "subject": "花无雪"
            },
            {
                "predicate": "歌手",
                "object_type": {"@value": "人物"},
                "subject_type": "歌曲",
                "object": {"@value": "泳儿"},
                "subject": "黛玉笑了"
            },
            {
                "predicate": "歌手",
                "object_type": {"@value": "人物"},
                "subject_type": "歌曲",
                "object": {"@value": "泳儿"},
                "subject": "花无雪"
            }
        ]
    }

    feature = convert_example_to_feature(example, tokenizer, label_map, cpe, max_length=320)
    tokens = feature.tokens
    labels = feature.labels
    print(tokens)
