import re
import sys
import json
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, AutoTokenizer, AutoConfig
from tqdm import tqdm
import pandas as pd
from loguru import logger


class NERmodel(nn.Module):
    def __init__(self, pretrained_model, hidden_size):
        super(NERmodel, self).__init__()
        self.pretrained_model = pretrained_model
        self.type2id = type2id
        self.id2type = {v: k for k, v in type2id.items()}
        self.bilstm = nn.LSTM(hidden_size, cfg.hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.start_linear = nn.Linear(hidden_size, 2)
        self.end_linear = nn.Linear(hidden_size, 2)
        self.type_linear = nn.Linear(cfg.hidden_dim*2, len(type2id))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, train=True):
        seq_out = self.pretrained_model(input_ids=x["input_ids"], attention_mask=x["attention_mask"])
        seq_out = seq_out.last_hidden_state
        seq_out = self.dropout(seq_out)
        # start label
        start_logits = self.start_linear(seq_out)
        # end label
        end_logits = self.end_linear(seq_out)
        seq_out, _ = self.bilstm(seq_out)
        # type label
        type_logits = self.type_linear(seq_out)
        outputs = {"start_logits":start_logits, "end_logits":end_logits, "type_logits":type_logits}
        if train:
            mask = x["attention_mask"].view(-1) == 1
            start_logits = start_logits.view(-1, start_logits.size()[2])[mask]
            end_logits = end_logits.view(-1, end_logits.size()[2])[mask]
            type_logits = type_logits.view(-1, type_logits.size()[2])[mask]
            loss_start = self.criterion(start_logits, x["start_label"].view(-1)[mask])
            loss_end = self.criterion(end_logits, x["end_label"].view(-1)[mask])
            loss_type = self.criterion(type_logits, x["type_label"].view(-1)[mask])
            # The sum of three types of losses is used as the model's loss.
            outputs["loss"] = loss_start + loss_end + loss_type
        return outputs
    
    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x, train=False)
            starts = torch.argmax(outputs["start_logits"], dim=2)
            ends = torch.argmax(outputs["end_logits"], dim=2)
            types = torch.argmax(outputs["type_logits"], dim=2)
        lens = (x["attention_mask"] == 1).sum(dim=1)
        res = []
        for start, end, type_label, ann_entity, word_offset, l in zip(starts, ends, types, x['ann_entity'], x['word_offset'], lens):
            start = [_.item() for _ in start[:l]]
            end = [_.item() for _ in end[:l]]
            type_label = [self.id2type[_.item()] for _ in type_label[:l]]
            pred_entity = self.get_pred_ents(start, end, type_label, word_offset)
            res.append({"ann_entity":ann_entity, "pred_entity":pred_entity, "start":start, "end":end, "type_label":type_label, 
                        "word_offset":word_offset})
        return res

    def get_pred_ents(self, start, end, type_label, word_offset):
        # Extract entities by predicted labels and word offsets.
        start_w, end_w, type_w = [], [], []
        # Convert token label into word label.
        for i in word_offset:
            if i[0]>=len(start): break
            s = 1 if sum(start[i[0]:i[1]+1])>0 else 0
            e = 1 if sum(end[i[0]:i[1]+1])>0 else 0
            # Calculate the frequency of token labels corresponding to each word, and select the most frequent one as the word label.
            t = Counter(type_label[i[0]:i[1]+1]).most_common(1)[0][0]
            start_w.append(s)
            end_w.append(e)
            type_w.append(t)
        i, n = 0, len(start_w)
        ents = []
        # Extract entities based on '1' in the start and end labels.
        while i<n:
            if start_w[i]==1:
                s = i
                for j in range(i, n):
                    if start_w[j]==1:
                        s = j
                    if end_w[j]==1:
                        e = j
                        t = Counter(type_w[s:e+1]).most_common(1)[0][0]
                        if t!='O':
                            ents.append((t, s, e))
                        i = j
                        break
                else:
                    break
            i += 1
        if not ents:
            # If no entity is extracted based on the start and end labels, extract entities using the type label instead.
            type_str = ''.join(map(str, [type2id[_] for _ in type_w]))
            match_res = re.finditer('[1-9]+', type_str)
            for i in match_res:
                s, e = i.span()
                t = id2type[int(i.group()[0])]
                if sum(start_w[s:e]+end_w[s:e])>0:
                    ents.append((t, s, e-1))
        ents_ = []
        for ent in ents:
            t, s, e = ent
            # Revise the boundaries of the entities based on the type label and certain rules.
            if s>1 and start_w[s-1]==start_w[s]==1 and end_w[s]==0 and type_w[s]==type_w[s-1]!='O':
                s = s-1
            if e+1<len(start_w) and end_w[e+1]==end_w[e]==1 and start_w[e+1]==0 and type_w[e]==type_w[e+1]!='O':
                e = e+1
            ents_.append((t, s, e))
        return ents_
    
class MyDataset(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        words = [_.lower() for _ in row.word]
        word_tokens = tokenizer(words, add_special_tokens=False).input_ids
        # Calculate word offset based on tokens
        input_ids, word_offset = [], []
        for i in word_tokens:
            word_offset.append((len(input_ids), len(input_ids)+len(i)-1))
            input_ids.extend(i)
        start_label, end_label, type_label = [0]*len(input_ids), [0]*len(input_ids), []
        for l, offset in zip(row.label, word_offset):
            if l[0] == "B":
                start_label[offset[0]] = 1
            elif l[0] == "E":
                end_label[offset[1]] = 1
            elif l[0] == "S":
                start_label[offset[0]] = 1
                end_label[offset[1]] = 1
            type_label.extend([l.split("-")[-1]]*(offset[1]-offset[0]+1))
        ann_entity = self.get_ann_ents(row.label)
        # Set the max_len to 200
        max_len = len(input_ids) if len(input_ids)<200 else 200
        type_label = [type2id[_] for _ in type_label[:max_len]]
        return {'input_ids':input_ids[:max_len], 'start_label':start_label[:max_len], 'end_label':end_label[:max_len], 
                'type_label':type_label[:max_len], 'ann_entity':ann_entity, 'word_offset':word_offset}
    
    def get_ann_ents(self, label):
        # Extract entities from sentences using labels and record the start and end word index of the entities.
        bio, types = [], []
        for l in label:
            l = l.split("-")
            bio.append(l[0])
            types.append(l[-1])
        match_res = re.finditer("BI*E|S", "".join(bio))
        ents = []
        for i in match_res:
            satrt, end = i.span()
            ent_type = types[satrt]
            if ent_type != "O":
                ents.append((ent_type, satrt, end-1))
        return ents

def collate_fn(batch):
    input_ids = pad_sequence([torch.LongTensor(_['input_ids']) for _ in batch], batch_first=True, padding_value=tokenizer.pad_token_id) # type: ignore
    attention_mask = (input_ids!=tokenizer.pad_token_id).int()
    start_label = pad_sequence([torch.LongTensor(_['start_label']) for _ in batch], batch_first=True, padding_value=0)
    end_label = pad_sequence([torch.LongTensor(_['end_label']) for _ in batch], batch_first=True, padding_value=0)
    type_label = pad_sequence([torch.LongTensor(_['type_label']) for _ in batch], batch_first=True, padding_value=type2id['O'])
    ann_entity = [_['ann_entity'] for _ in batch]
    word_offset = [_['word_offset'] for _ in batch]
    return {'input_ids':input_ids, 'attention_mask':attention_mask, 'start_label':start_label, 'end_label':end_label,
            'type_label':type_label, 'ann_entity':ann_entity, 'word_offset':word_offset}

def make_dataloader():
    df = pd.read_parquet(cfg.data_file)
    train_df, valid_df, test_df = df.loc[(df.type=="train")|(df.type=="data_aug")], df[df.type=="valid"], df[df.type=="test"]
    logger.info(f'train size:{len(train_df)}, valid size:{len(valid_df)}, test size:{len(test_df)}')
    trainset = MyDataset(train_df)
    trainloader = DataLoader(trainset, batch_size=cfg.batch_size, collate_fn=collate_fn, shuffle=True)
    validset = MyDataset(valid_df)
    validloader = DataLoader(validset, batch_size=cfg.batch_size_pred, collate_fn=collate_fn, shuffle=False)
    testset =  MyDataset(test_df)
    testloader = DataLoader(testset, batch_size=cfg.batch_size_pred, collate_fn=collate_fn, shuffle=False)
    return trainloader, validloader, testloader

def predict(model, dataloader):
    model.eval()
    preds, n_correct, n_y_true, n_y_pred = [], 0, 0, 0
    for batch in dataloader:
        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device)
        pred = model.predict(batch)
        for d in pred:
            preds.append(d)
            ann_ents = d["ann_entity"]
            pred_ents = d["pred_entity"]
            tp = [e for e in pred_ents if e in ann_ents]
            n_correct += len(tp)
            n_y_true += len(ann_ents)
            n_y_pred += len(pred_ents)
    p = 0 if n_y_pred == 0 else n_correct / n_y_pred
    r = 0 if n_y_true == 0 else n_correct / n_y_true
    f1 = 0 if p + r == 0 else 2 * p * r / (p + r)
    return preds, p, r, f1

def run(pretrained_model, hidden_size):
    trainloader, validloader, testloader = make_dataloader()
    model = NERmodel(pretrained_model, hidden_size)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    logger.info("start training...")
    f1_tmp = 0
    for epoch in range(cfg.epochs):
        model.train()
        loop = tqdm(trainloader, desc=f"Epoch [{epoch+1}/{cfg.epochs}]")
        for batch in loop:
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device)
            outputs = model(batch, train=True)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loop.set_postfix(loss=loss.item())
        preds, p, r, f1 = predict(model, validloader)
        logger.info(f"epoch:{epoch + 1}, valid P/R/F1: {p:.4f}/{r:.4f}/{f1:.4f}")
        # When there is an improvement in F1 value, save the checkpoint.
        if f1 > f1_tmp:
            f1_tmp = f1
            torch.save(model.state_dict(), cfg.model_save_path)
    # Load the checkpoint with the highest F1 score on the valid set for predicting the test set.
    model.load_state_dict(torch.load(cfg.model_save_path))
    preds, p, r, f1 = predict(model, testloader)
    df_ = pd.DataFrame(preds)
    df_.to_csv('test-1.csv')
    logger.info(f"test P/R/F1: {p:.4f}/{r:.4f}/{f1:.4f}")

class Config:
    def __init__(self, **kws):
        self.seed = 5
        self.batch_size = 16
        self.batch_size_pred = 128
        self.lr = 2e-5
        self.epochs = 10
        self.data_file = "./data/mdmt.parquet"
        self.model_path = "scibert_scivocab_uncased"
        self.model_save_path = "./001.pth"
        self.hidden_dim = 512
        self.log_path = "./log.txt"
        self.__dict__.update(kws)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
type2id = {"O":0, "Method":1, "Dataset":2, "Metric":3, "Tool":4}
id2type = {v: k for k, v in type2id.items()}

cfg = Config()
torch.manual_seed(cfg.seed)  # Set random seed.
logger.remove(handler_id=None)
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {message}")
logger.add(cfg.log_path, encoding="utf-8", format="{time:YYYY-MM-DD HH:mm:ss} | {message}")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)
hidden_size = AutoConfig.from_pretrained(cfg.model_path).hidden_size

logger.info(json.dumps(cfg.__dict__, indent=4, ensure_ascii=False))
pretrained_model = BertModel.from_pretrained(cfg.model_path, )
run(pretrained_model, hidden_size)

# for i in range(1, 6):
#     cfg = Config(lr=i*1e-5)
#     logger.info(json.dumps(cfg.__dict__, indent=4, ensure_ascii=False))
#     pretrained_model = BertModel.from_pretrained(cfg.model_path)
#     run(pretrained_model, hidden_size)
