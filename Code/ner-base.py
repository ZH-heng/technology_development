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
from torchcrf import CRF
from tqdm import tqdm
import pandas as pd
from loguru import logger


class NERmodel(nn.Module):
    def __init__(self, pretrained_model, hidden_size):
        super(NERmodel, self).__init__()
        self.pretrained_model = pretrained_model
        self.label_linear = nn.Linear(hidden_size, len(label2id))
        self.dropout = nn.Dropout(0.2)
        self.criterion = nn.CrossEntropyLoss()
        self.crf = CRF(len(label2id), batch_first=True)
        self.use_crf = cfg.use_crf

    def forward(self, x, train=True):
        seq_out = self.pretrained_model(input_ids=x['input_ids'], attention_mask=x['attention_mask'])
        seq_out = seq_out.last_hidden_state
        seq_out = self.dropout(seq_out)
        logits = self.label_linear(seq_out)
        outputs = {'logits':logits}
        if train:
            # Whether to use CRF.
            if cfg.use_crf:
                mask = x['attention_mask'] == 1
                loss = self.crf(logits, x['token_label'], mask) * (-1)
            else:
                mask = x['attention_mask'].view(-1) == 1
                logits = logits.view(-1, logits.size()[2])[mask]
                label = x['token_label'].view(-1)[mask]
                loss = self.criterion(logits, label)
            outputs['loss'] = loss
        return outputs

    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x, train=False)
            pred = torch.argmax(outputs['logits'], dim=2)
        lens = (x['attention_mask'] == 1).sum(dim=1)
        res = []
        for i, word_offset, l, ann_entity in zip(pred, x['word_offset'], lens, x['ann_entity']):
            pred_label = [id2label[_.item()] for _ in i[:l]]
            pred_entity = self.get_pred_ents(pred_label, word_offset)
            res.append({'ann_entity':ann_entity, 'pred_entity':pred_entity, 'pred_label':pred_label, 'word_offset':word_offset})
        return res

    def get_pred_ents(self, pred_label, word_offset):
        # Extract entities by predicted token labels and word offsets.
        bio, types = [], []
        for l in pred_label:
            l = l.split('-')
            bio.append(l[0])
            types.append(l[-1])
        match_res = re.finditer('BI*E|S', ''.join(bio))
        ents = []
        for i in match_res:
            satrt, end = i.span()
            ent_type = Counter(types[satrt:end]).most_common(1)[0][0]
            if ent_type != 'O':
                ents.append((ent_type, satrt, end-1))
        # Convert token index to word index.
        token2word = {}
        for i, offset in enumerate(word_offset):
            for _ in range(offset[0], offset[1]+1):
                token2word[_] = i
        ents = [(_[0], token2word[_[1]], token2word[_[2]]) for _ in ents]
        return ents

class MyDataset(Dataset):
    def __init__(self, df, label2id):
        self.df = df
        self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Tokenize the sentence, obtain the mapping relationship between words and tokens positions, and construct token_label.
        row = self.df.iloc[index]
        words = [_.lower() for _ in row.word]
        tokens = tokenizer(words, add_special_tokens=False).input_ids
        input_ids, word_offset = [], []
        for token in tokens:
            word_offset.append((len(input_ids), len(input_ids)+len(token)-1))
            input_ids.extend(token)
        word_label, token_label = row.label, []
        # Convert word labels to token labels.
        for l, offset in zip(word_label, word_offset):
            if l[0]=='B':
                token_label.append(l)
                token_label.extend([f'I{l[1:]}']*(offset[1]-offset[0]))
            elif l[0]=='E':
                token_label.extend([f'I{l[1:]}']*(offset[1]-offset[0]))
                token_label.append(l)
            elif l[0]=='S':
                if offset[0]==offset[1]:
                    token_label.append(l)
                else:
                    token_label.append(f'B{l[1:]}')
                    token_label.extend([f'I{l[1:]}']*(offset[1]-offset[0]-1))
                    token_label.append(f'E{l[1:]}')
            else:
                token_label.extend([l]*(offset[1]-offset[0]+1))
        ann_entity = self.get_ann_ents(word_label)
        # Set the max_len to 200
        max_len = len(input_ids) if len(input_ids)<200 else 200
        token_label = [self.label2id[_] for _ in token_label[:max_len]]
        return {'input_ids':input_ids[:max_len], 'token_label':token_label, 'ann_entity':ann_entity, 'word_offset':word_offset}
        
    def get_ann_ents(self, label):
        # Extract entities from sentences using labels and record the start and end word index of the entities.
        bio, types = [], []
        for l in label:
            l = l.split('-')
            bio.append(l[0])
            types.append(l[-1])
        match_res = re.finditer('BI*E|S', ''.join(bio))
        ents = []
        for i in match_res:
            satrt, end = i.span()
            ent_type = types[satrt]
            if ent_type != 'O':
                ents.append((ent_type, satrt, end-1))
        return ents

def collate_fn(batch):
    input_ids = pad_sequence([torch.LongTensor(_['input_ids']) for _ in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids!=tokenizer.pad_token_id).int()
    token_label = pad_sequence([torch.LongTensor(_['token_label']) for _ in batch], batch_first=True, padding_value=0)
    ann_entity = [_['ann_entity'] for _ in batch]
    word_offset = [_['word_offset'] for _ in batch]
    return {'input_ids':input_ids, 'attention_mask':attention_mask, 'token_label':token_label, 'ann_entity':ann_entity, 'word_offset':word_offset}

def make_dataloader():
    # Preprocess the trainset, validset, and testset.
    df = pd.read_parquet(cfg.data_file)
    train_df, valid_df, test_df = df.loc[(df.type=="train")|(df.type=="data_aug")], df[df.type=="valid"], df[df.type=="test"]
    logger.info(f'train size:{len(train_df)}, valid size:{len(valid_df)}, test size:{len(test_df)}')
    trainset = MyDataset(train_df, label2id)
    trainloader = DataLoader(trainset, batch_size=cfg.batch_size, collate_fn=collate_fn, shuffle=True)
    validset = MyDataset(valid_df, label2id)
    validloader = DataLoader(validset, batch_size=cfg.batch_size_pred, collate_fn=collate_fn, shuffle=False)
    testset =  MyDataset(test_df, label2id)
    testloader = DataLoader(testset, batch_size=cfg.batch_size_pred, collate_fn=collate_fn, shuffle=False)
    return trainloader, validloader, testloader

def predict(model, dataloader):
    model.eval()
    n_correct, n_y_true, n_y_pred = 0, 0, 0
    for batch in dataloader:
        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device)
        pred = model.predict(batch)
        for d in pred:
            ann_ents = d['ann_entity']
            pred_ents = d['pred_entity']
            tp = [e for e in pred_ents if e in ann_ents]
            n_correct += len(tp)
            n_y_true += len(ann_ents)
            n_y_pred += len(pred_ents)
    p = 0 if n_y_pred == 0 else n_correct / n_y_pred
    r = 0 if n_y_true == 0 else n_correct / n_y_true
    f1 = 0 if p + r == 0 else 2 * p * r / (p + r)
    return p, r, f1

def run(pretrained_model, hidden_size):
    trainloader, validloader, testloader = make_dataloader()
    model = NERmodel(pretrained_model, hidden_size)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    logger.info('start training...')
    f1_tmp = 0
    for epoch in range(cfg.epochs):
        model.train()
        loop = tqdm(trainloader, desc=f'Epoch [{epoch+1}/{cfg.epochs}]')
        for batch in loop:
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device)
            outputs = model(batch, train=True)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loop.set_postfix(loss=loss.item())
        p, r, f1 = predict(model, validloader)
        logger.info(f'epoch:{epoch + 1}, valid: P/R/F1: {p:.4f}/{r:.4f}/{f1:.4f}')
        # When there is an improvement in F1 value, save the checkpoint.
        if f1 > f1_tmp:
            f1_tmp = f1
            torch.save(model.state_dict(), cfg.model_save_path)
    # Load the checkpoint with the highest F1 score on the valid set for predicting the test set.
    model.load_state_dict(torch.load(cfg.model_save_path))
    p, r, f1 = predict(model, testloader)
    logger.info(f'test: P/R/F1: {p:.4f}/{r:.4f}/{f1:.4f}')

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
        self.use_crf = False
        self.log_path = './log.txt'
        self.__dict__.update(kws)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label2id = {'O': 0, 'B-Method': 1, 'I-Method': 2, 'E-Method': 3, 'S-Method': 4, 'B-Dataset': 5, 'I-Dataset': 6, 'E-Dataset': 7, 'S-Dataset': 8, 'B-Metric': 9, 'I-Metric': 10, 'E-Metric': 11, 'S-Metric': 12, 'B-Tool': 13, 'I-Tool': 14, 'E-Tool': 15, 'S-Tool': 16}
id2label = {v: k for k, v in label2id.items()}

cfg = Config()
torch.manual_seed(cfg.seed)  # Set random seed.
logger.remove(handler_id=None)
logger.add(sys.stderr, format='{time:YYYY-MM-DD HH:mm:ss} | {message}')
logger.add(cfg.log_path, encoding='utf-8', format='{time:YYYY-MM-DD HH:mm:ss} | {message}')
tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)
hidden_size = AutoConfig.from_pretrained(cfg.model_path).hidden_size

logger.info(json.dumps(cfg.__dict__, indent=4, ensure_ascii=False))
pretrained_model = BertModel.from_pretrained(cfg.model_path)
run(pretrained_model, hidden_size)

# for i in range(1, 6):
#     cfg = Config(lr=i*1e-5)
#     logger.info(json.dumps(cfg.__dict__, indent=4, ensure_ascii=False))
#     pretrained_model = BertModel.from_pretrained(cfg.model_path)
#     run(pretrained_model, hidden_size)
