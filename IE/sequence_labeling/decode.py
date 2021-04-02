# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2021/3/30
@function: 预测结果解码
"""
import os
import re
import json
import numpy as np
import pickle as pkl


multi_relation= {
    "配音":{"inWork": "影视作品", "@value": "人物"},
    "上映时间":{"inArea": "地点", "@value": "Date"},
    "票房":{"inArea": "地点", "@value": "Number"},
    "获奖":{"inWork": "作品", "onDate": "Date", "@value": "奖项", "period": "Number"},
    "饰演":{"inWork": "影视作品", "@value": "人物"}
}


def decoding(features, predict_logits, label_map):
    result=[]
    for i, feature in enumerate(features): # sentence level
        tokens=feature.tokens
        token_to_origin_start_index=feature.token_to_origin_start_index
        token_to_origin_end_index=feature.token_to_origin_end_index
        raw_text=feature.raw_text
        predict_logit=predict_logits[i]

        flag=None
        token_pos_tags={}
        left, right=0, 0
        while left<len(tokens) and right<len(tokens):
            if tokens[right]=="[PAD]":
                break
            # 获得当前token的label
            tag_idx=predict_logit[right]
            tags=[]
            for j, idx in enumerate(tag_idx):
                if idx==1:
                    tags.append(label_map[str(j)])
            # 非O的头指针和尾指针是一样的
            if len(tags)==1 and tags[0]=="O":
                if flag:
                    entity=""
                    for index in range(left, right):
                        token_start_index=token_to_origin_start_index[index]
                        token_end_index=token_to_origin_end_index[index]
                        entity+=raw_text[token_start_index:token_end_index+1]
                    # entity="".join(tokens[left:right])
                    for tag in flag:
                        token_pos_tags[tag]=token_pos_tags.get(tag,[])
                        token_pos_tags[tag].append(entity)
                    flag=None
                    left=right
                else:
                    left+=1
                    right+=1
            else:
                if not flag:
                    flag=tags
                right+=1

        spo_list= []
        for pos_tag in token_pos_tags:
            try:
                sub_obj, predicate, obj_type=pos_tag.split("-")
            except ValueError:
                continue
            if sub_obj != "S":
                continue
            obj_pos_tag = "-".join(["O", predicate, obj_type])
            sub_values = token_pos_tags[pos_tag]
            try:
                obj_values = token_pos_tags[obj_pos_tag]
            except KeyError:
                continue
            if predicate in multi_relation:
                for spo in spo_list:
                    if predicate==spo["predicate"]:
                        for key, val in multi_relation[predicate].items():
                            if obj_type==val:
                                spo["object_type"][key]=val
                                spo["object"]=obj_values[0]
                else:
                    for sub_value in sub_values:
                        for obj_value in obj_values:
                            for key, val in multi_relation[predicate].items():
                                if obj_type==val:
                                    spo = {"predicate": predicate,
                                           "subject_type": "XX",
                                           "subject": sub_value,
                                           "object_type": {"@value": obj_type},
                                           "object": {"@value": obj_value}}
                                    spo_list.append(spo)

            else:
                for sub_value in sub_values:
                    for obj_value in obj_values:
                        spo = {"predicate": predicate,
                               "subject_type": "XX",
                               "subject": sub_value,
                               "object_type": {"@value": obj_type},
                               "object": {"@value": obj_value}}
                        spo_list.append(spo)

        result.append(spo_list)
    return result


















if __name__ == '__main__':
    label_map, id2label=json.load(open("../origin_data/labels2idx.json", encoding="utf8"))
    predict_logits, features=pkl.load(open("./test.pkl","rb"))

    predict_logits[predict_logits>=0.5]=1
    predict_logits[predict_logits<0.5]=0

    decoding(features, predict_logits.astype(np.int).tolist(), id2label)

