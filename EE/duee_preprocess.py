# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2021/3/30
@function: 事件抽取-句子级别预处理
"""
import os
import json
from tqdm import tqdm


def get_label(labels, start_index, length, _type):
    for i in range(start_index, start_index+length):
        suffix="B-" if i==start_index else "I-"
        labels[i]=f"{suffix}{_type}"
    return labels


def data_process(filepath, model="trigger",training=True):
    sentences=[]
    output=["text_a\tlabel"] if training else ["text_a"]
    with open(filepath, encoding="utf8") as fn:
        lines=fn.readlines()
        for line in tqdm(lines):
            item=json.loads(line.strip())
            _id=item["id"]
            text_a=["，" if t==" " or t=="\n" or t=="\t" else t for t in list(item["text"].lower())]

            if training:
                if model=="trigger":
                    labels=["O"] * len(text_a)
                    for event in item.get("event_list",[]):
                        event_type=event["event_type"]
                        start=event["trigger_start_index"]
                        trigger=event["trigger"]
                        labels=get_label(labels, start, len(trigger),event_type)
                    output.append("{}\t{}".format("\002".join(text_a), "\002".join(labels)))

                elif model=="role":
                    for event in item.get("event_list",[]):
                        labels=["O"]*len(text_a)
                        for arg in event["arguments"]:
                            role_type=arg["role"]
                            argument=arg["argument"]
                            start=arg["argument_start_index"]
                            labels=get_label(labels, start, len(argument), role_type)
                        output.append("{}\t{}".format("\002".join(text_a), "\002".join(labels)))

            else:
                sentences.append({"text":item["text"],"id":_id})
                output.append("\002".join(text_a))
    return output


def schema_process(path, model="trigger"):
    labels=["O"]
    def label_add(labels, _type):
        if "B-{}".format(_type) not in labels:
            labels.extend(["B-{}".format(_type), "I-{}".format(_type)])
        return labels
    with open(path, encoding="utf8") as fn:
        for line in fn:
            item=json.loads(line.strip())
            if model=="trigger":
                labels=label_add(labels, item["event_type"])
            elif model=="role":
                for role in item["role_list"]:
                    labels=label_add(labels, role["role"])
    tags= {}
    for index, label in enumerate(labels):
        tags[label]=index
    return tags




if __name__ == '__main__':

    def write_to_file(to_file, data):
        with open(to_file, "w",encoding="utf8") as fn:
            for line in data:
                fn.write(line+"\n")
    print("Start schema process...")
    schema_path="./origin_data/duee_event_schema.json"
    tags_trigger_path="./origin_data/duee_event_trigger_tag.json"
    tags_role_path="./origin_data/duee_event_role_tag.json"
    tags_trigger=schema_process(schema_path)
    tags_role=schema_process(schema_path, model="role")
    json.dump(tags_trigger, open(tags_trigger_path,"w",encoding="utf8"),ensure_ascii=False, indent=2)
    json.dump(tags_role, open(tags_role_path,"w",encoding="utf8"),ensure_ascii=False, indent=2)
    print("End schema process...\n")

    print("Start data process...")
    trigger_save_dir="./origin_data/duee_trigger"
    role_save_dir="./origin_data/duee_role"
    if not os.path.exists(trigger_save_dir):
        os.makedirs(trigger_save_dir)
    if not os.path.exists(role_save_dir):
        os.makedirs(role_save_dir)

    train_path="./origin_data/duee_train.json"
    dev_path="./origin_data/duee_dev.json"
    test_path="./origin_data/duee_test1.json"

    train_trigger=data_process(train_path, "trigger")
    dev_trigger=data_process(dev_path, "trigger")
    test_trigger=data_process(test_path, "trigger")

    train_role=data_process(train_path, "role")
    dev_role=data_process(dev_path, "role")
    test_role=data_process(test_path, "role")

    write_to_file(f"{trigger_save_dir}/train.tsv", train_trigger)
    write_to_file(f"{trigger_save_dir}/dev.tsv", dev_trigger)
    write_to_file(f"{trigger_save_dir}/test.tsv", test_trigger)

    write_to_file(f"{role_save_dir}/train.tsv", train_role)
    write_to_file(f"{role_save_dir}/dev.tsv", dev_role)
    write_to_file(f"{role_save_dir}/test.tsv", test_role)
    print("End data process")







