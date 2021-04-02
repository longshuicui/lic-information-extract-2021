# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2021/3/19
@function: 数据预处理操作
"""
import json
from collections import defaultdict


def get_labels():
    predicate2idx={}
    with open("../origin_data/schema.json", encoding="utf8") as file:
        for line in file:
            item=json.loads(line.strip())
            predicate=item["predicate"]
            object_type=item["object_type"]
            for key, value in object_type.items():
                label=predicate+"-"+value
                if label not in predicate2idx:
                    predicate2idx[label]=len(predicate2idx)
    labels2idx={"O":0, "I":1}
    for t in ["S-","O-"]:
        for predicate in predicate2idx:
            label=t+predicate
            labels2idx[label]=len(labels2idx)

    idx2labels={val:key for key, val in labels2idx.items()}

    json.dump([labels2idx, idx2labels], open("../origin_data/labels2idx_last_year.json", "w", encoding="utf8"), ensure_ascii=False, indent=2)


def get_length():
    lengths=[]
    with open("../origin_data/duie_train.json", encoding="utf8") as fn:
        for line in fn:
            item=json.loads(line.strip())
            text=item["text"]
            lengths.append(len(text))
    print(max(lengths))
    print(min(lengths))
    print(sum(lengths)/len(lengths))




## 256+128=384  256+64=320

get_length()