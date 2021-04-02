# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2021/3/20
@function:
"""
import time
import argparse
import random
import pickle as pkl
from transformers import AdamW, get_linear_schedule_with_warmup, BertConfig
from model import BertForTokenClassification
from load_data_pos import *
from extract_chinese_and_punts import ChineseAndPunctuationExtractor
from decode import decoding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="hfl/chinese-roberta-wwm-ext", help="The pretrain model from huggingface")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="epochs")
    parser.add_argument("--max_length", type=int, default=320, help="the sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="the sequence length")
    args = parser.parse_args()
    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def metric(logits, labels):
    """
    计算F1
    :param logits: [batch_size, seq_len, num_lables]
    :param labels: [batch_size, seq_len, num_lables]
    :return: p,r,f
    """
    logits=logits.view(-1, logits.size(-1))
    logits=torch.sigmoid(logits)
    labels=labels.view(-1, labels.size(-1))
    mid=torch.where(logits==labels, torch.ones(logits.size(0)), torch.zeros(logits.size(0)))
    acc=torch.sum(mid)/len(mid)
    return acc.item()


def do_train(args):
    device="cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained(args.name)
    cpe = ChineseAndPunctuationExtractor()
    label_map, _ = json.load(open("../origin_data/labels2idx.json", encoding="utf8"))
    train_features = from_file("../origin_data/duie_train.json", tokenizer, label_map, cpe, args.max_length)
    dev_features = from_file("../origin_data/duie_dev.json", tokenizer, label_map, cpe, args.max_length)
    counts = len(train_features)
    logger.info(f"Train dataset size: {counts}, Dev dataset size: {len(dev_features)}")
    if len(train_features)%args.batch_size==0:
        one_epoch_steps = len(train_features) // args.batch_size
    else:
        one_epoch_steps = len(train_features) // args.batch_size + 1
    total_steps = one_epoch_steps*args.epochs
    logger.info(f"Training step: {total_steps}")

    bert = BertForTokenClassification.from_pretrained(args.name, num_labels=len(label_map)).to(device)
    optimizer = AdamW(params=bert.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    best_eval_f1=0
    min_eval_loss=float("inf")
    for epoch in range(args.epochs):
        train_generator = batch_generator(train_features, args.batch_size)
        valid_generator = batch_generator(dev_features, args.batch_size, False)
        logger.info(f"======== Epoch {epoch + 1:} / {args.epochs:} ========")
        logger.info("Training...")
        bert.train()
        start_train=time.time()
        total_train_loss=0
        for step, batch in enumerate(train_generator):
            batch_input_ids = batch[0].to(device=device)
            batch_input_mask = batch[1].to(device=device)
            batch_type_ids = batch[2].to(device=device)
            batch_labels = batch[3].to(device=device)
            outputs=bert(batch_input_ids, batch_input_mask, batch_type_ids, labels=batch_labels)
            bert.zero_grad()
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss+=outputs.loss.item()

            if step%100==0:
                logger.info(f"  Step: {step+1:>5}/{one_epoch_steps:>1}, current loss: {outputs.loss.item():.6f}")

        average_train_loss=total_train_loss/(step+1)
        trainingtime=time.time()-start_train
        logger.info(f"  Average training BCELoss: {average_train_loss:.6f}; Take time: {trainingtime:.3f}")

        logger.info("Running Validation...")
        bert.eval()
        start_eval=time.time()
        total_eval_loss=0
        total_eval_f1=0
        for step, batch in enumerate(valid_generator):
            batch_input_ids = batch[0].to(device=device)
            batch_input_mask = batch[1].to(device=device)
            batch_type_ids = batch[2].to(device=device)
            batch_labels = batch[3].to(device=device)
            with torch.no_grad():
                outputs=bert(batch_input_ids, batch_input_mask, batch_type_ids, labels=batch_labels)
                total_eval_loss+=outputs.loss.item()
                # total_eval_f1+=metric(outputs.logits, batch_labels)
        average_eval_loss=total_eval_loss/(step+1)
        # average_eval_f1=total_eval_f1/(step+1)
        validation_time=time.time()-start_eval
        logger.info(f"  Average eval BCELoss: {average_eval_loss:.6f}; Take time: {validation_time:.3f}")

        # if average_eval_f1>best_eval_f1:
        #     best_eval_f1=average_eval_f1
        #     logger.info("   Save model...")
        #     torch.save(bert.state_dict(),f"model_{epoch}.pt")
        if average_eval_loss<min_eval_loss:
            min_eval_loss=average_eval_loss
            logger.info("  Save model...")
            torch.save(bert.state_dict(),f"model_{epoch}.pt")


def do_infer(args):
    device="cuda" if torch.cuda.is_available() else "cpu"
    config=BertConfig.from_pretrained(args.name)
    tokenizer = BertTokenizer.from_pretrained(args.name)
    cpe = ChineseAndPunctuationExtractor()
    label_map, id2label = json.load(open("../origin_data/labels2idx.json", encoding="utf8"))
    config.num_labels=len(label_map)
    features = from_file("../origin_data/duie_test1.json", tokenizer, label_map, cpe, args.max_length)
    bert = BertForTokenClassification.from_pretrained("./model_2.pt", config=config).to(device)
    infer_generator=batch_generator(features, args.batch_size, False)
    predict_logits=[]
    for step, batch in enumerate(infer_generator):
        batch_input_ids = batch[0].to(device=device)
        batch_input_mask = batch[1].to(device=device)
        batch_type_ids = batch[2].to(device=device)
        outputs=bert(batch_input_ids, batch_input_mask, batch_type_ids)
        batch_predict=torch.sigmoid(outputs.logits).cpu().detach().numpy()
        predict_logits.append(batch_predict)

    predict_logits=np.concatenate(predict_logits, axis=0)
    assert predict_logits.shape[0]==len(features)
    predict_logits[predict_logits >= 0.3] = 1
    predict_logits[predict_logits < 0.3] = 0
    predict_logits=predict_logits.astype(np.int).tolist()
    result=decoding(features, predict_logits, id2label)
    with open("result.json","w",encoding="utf8") as outp:
        for line in result:
            outp.write(json.dumps(line, ensure_ascii=False)+"\n")












if __name__ == '__main__':
    setup_seed(1)
    args = add_arguments()
    # do_train(args)
    do_infer(args)
    # exit()