"""Дообучение cross-encoder'а на парах товаров.

    python train_ce.py <model_name> <tag> [epochs] [batch] [max_length]

Вход: /tmp/matching/items_human_clean.parquet (чистые тексты) + matches.parquet.
Сплит item-level: товар не попадает одновременно в train и val.
Метрика: macro PR-AUC по категориям — как у zero-shot бейзлайна (0.3756).
"""
import hashlib
import json
import math
import os
import sys
import time

os.environ.setdefault('HF_HOME', '/tmp/hf')

import numpy as np
import polars as pl
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL = sys.argv[1]
TAG = sys.argv[2]
EPOCHS = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
BATCH = int(sys.argv[4]) if len(sys.argv) > 4 else 32
MAX_LENGTH = int(sys.argv[5]) if len(sys.argv) > 5 else 384

DATA = '/home/jovyan/xakaton/data'
OUT = f'/tmp/matching/ce_{TAG}'
LR = 2e-5
SEED = 42
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# --- данные ----------------------------------------------------------------
items = pl.read_parquet('/tmp/matching/items_human_clean.parquet')
matches = pl.read_parquet(f'{DATA}/matches.parquet')
text = dict(zip(items['id'].to_list(), items['text'].to_list()))
category = dict(zip(items['id'].to_list(), items['category'].to_list()))


def side(item_id, val_frac=0.2, seed=SEED):
    h = hashlib.blake2b(f'{seed}:{item_id}'.encode(), digest_size=8).digest()
    return int.from_bytes(h, 'big') / 2**64 < val_frac


id1 = matches['id1'].to_list()
id2 = matches['id2'].to_list()
y = matches['target'].to_numpy().astype(np.float32)

v1 = np.array([side(i) for i in id1])
v2 = np.array([side(i) for i in id2])
is_val = v1 & v2
is_train = ~v1 & ~v2
log(f'train {is_train.sum()}, val {is_val.sum()}, отброшено смешанных {(~is_train & ~is_val).sum()}')


class Pairs(Dataset):
    def __init__(self, mask, swap):
        self.idx = np.flatnonzero(mask)
        self.swap = swap

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        a, b = text[id1[j]], text[id2[j]]
        if self.swap and np.random.rand() < 0.5:
            a, b = b, a          # матчинг симметричен, а [SEP] — нет
        return a, b, y[j]


tokenizer = AutoTokenizer.from_pretrained(MODEL)


def collate(batch):
    a, b, labels = zip(*batch)
    enc = tokenizer(list(a), list(b), padding=True, truncation=True,
                    max_length=MAX_LENGTH, return_tensors='pt')
    enc['labels'] = torch.tensor(labels)
    return enc


train_ds, val_ds = Pairs(is_train, swap=True), Pairs(is_val, swap=False)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate,
                      num_workers=4, drop_last=True, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH * 2, shuffle=False, collate_fn=collate,
                    num_workers=4, pin_memory=True)

# длина пар в токенах
probe = [collate([train_ds[i] for i in np.random.randint(0, len(train_ds), 2000)])]
log(f'длина пары в токенах при max_length={MAX_LENGTH}: {probe[0]["input_ids"].shape}')

# --- модель ----------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=1, dtype=torch.float32).cuda()
model.gradient_checkpointing_enable()
log(f'{MODEL}: {sum(p.numel() for p in model.parameters())/1e6:.0f}M параметров')

steps = int(len(train_dl) * EPOCHS)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
loss_fn = torch.nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler('cuda')


@torch.inference_mode()
def evaluate():
    model.eval()
    scores = []
    for batch in val_dl:
        labels = batch.pop('labels')
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(**batch).logits.squeeze(-1)
        scores.append(out.float().cpu().numpy())
    model.train()
    return np.concatenate(scores)


def macro_pr_auc(y_true, score, groups, min_size=50):
    per = {}
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= min_size and 0 < y_true[m].sum() < m.sum():
            per[g] = float(average_precision_score(y_true[m], score[m]))
    return float(np.mean(list(per.values()))), per


val_y = y[is_val]
val_cat = np.array([category[i] for i in np.array(id1)[is_val]])

# --- обучение --------------------------------------------------------------
log(f'старт: {steps} шагов, batch={BATCH}, lr={LR}, max_length={MAX_LENGTH}')
model.train()
step = 0
t0 = time.time()
running = 0.0
best = -1.0

done = False
while not done:
    for batch in train_dl:
        labels = batch.pop('labels').cuda(non_blocking=True)
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(**batch).logits.squeeze(-1)
            loss = loss_fn(logits.float(), labels)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        running += loss.item()
        step += 1
        if step % 200 == 0:
            rate = step / (time.time() - t0)
            log(f'  шаг {step}/{steps} loss={running/200:.4f} '
                f'({rate:.2f} it/s, осталось {(steps-step)/rate/60:.0f} мин)')
            running = 0.0

        if step % 2000 == 0 or step == steps:
            scores = evaluate()
            macro, per_cat = macro_pr_auc(val_y, scores, val_cat)
            ap = float(average_precision_score(val_y, scores))
            roc = float(roc_auc_score(val_y, scores))
            log(f'  ОЦЕНКА шаг {step}: MACRO PR-AUC={macro:.4f} PR-AUC={ap:.4f} ROC-AUC={roc:.4f}')
            if macro > best:
                best = macro
                np.save(f'{OUT}/val_scores.npy', scores)
                json.dump({'model': MODEL, 'step': step, 'macro_pr_auc': macro,
                           'pr_auc': ap, 'roc_auc': roc, 'per_category': per_cat,
                           'max_length': MAX_LENGTH, 'batch': BATCH, 'lr': LR},
                          open(f'{OUT}/metrics.json', 'w'), ensure_ascii=False, indent=2)
                model.save_pretrained(f'{OUT}/best')
                tokenizer.save_pretrained(f'{OUT}/best')
                log(f'  сохранено (лучший macro={best:.4f})')

        if step >= steps:
            done = True
            break

log(f'ФИНИШ {TAG}: лучший MACRO PR-AUC = {best:.4f} (zero-shot косинус был 0.3756)')
