"""Эксперимент: предобучение на LLM-парах -> дообучение на человеческих.

Проблема, которую лечим: модель, обученная только на 292k человеческих пар,
запоминает конкретные товары. На незнакомых товарах метрика падает
0.7750 -> 0.5927, а на лидерборде 0.4454.

LLM-набор даёт 11.2M пар и 12.4M других товаров — то самое разнообразие.
Метки мягкие (доля голосов 0..1), BCE это переваривает напрямую.

Меряем на ДВУХ наборах:
  human val    — та же разметка, что в тесте (но знакомые товары), опорное 0.7750
  llm holdout  — незнакомые товары, честный прокси лидерборда, опорное 0.5927
"""
import hashlib
import json
import os
import sys
import time

os.environ.setdefault('HF_HOME', '/tmp/hf')

import numpy as np
import polars as pl
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

sys.path.insert(0, '/home/jovyan/xakaton')
from preprocess import preprocess_item

DATA = '/home/jovyan/xakaton/data'
OUT = '/tmp/matching/ce_llm'
BASE_MODEL = 'BAAI/bge-reranker-v2-m3'
N_LLM_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 600_000
BATCH = 64
MAX_LENGTH = 384
LR_STAGE1 = 2e-5
LR_STAGE2 = 1e-5
SEED = 42
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


# ---------------------------------------------------------------- данные
log('готовлю LLM-holdout (тот же, что дал 0.5927)')
llm = pl.read_parquet(f'{DATA}/matches_llm.parquet')
conf = llm.filter((pl.col('target') == 0.0) | (pl.col('target') == 1.0))
holdout = conf.sample(60_000, seed=1)
hold_items = set(holdout['id1'].to_list()) | set(holdout['id2'].to_list())
log(f'holdout: {holdout.height} пар, {len(hold_items)} товаров')

# train-пары не должны касаться товаров из holdout — иначе прокси сломается
rest = llm.filter(~(pl.col('id1').is_in(list(hold_items)) | pl.col('id2').is_in(list(hold_items))))
llm_train = rest.sample(min(N_LLM_TRAIN, rest.height), seed=2)
log(f'LLM train: {llm_train.height} пар (из {rest.height} доступных)')

human = pl.read_parquet(f'{DATA}/matches.parquet')
h1, h2 = human['id1'].to_list(), human['id2'].to_list()
hy = human['target'].to_numpy().astype(np.float32)

parent = {}


def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for a, b in zip(h1, h2):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb


def hsh(r, seed=SEED):
    return int.from_bytes(hashlib.blake2b(f'{seed}:{r}'.encode(), digest_size=8).digest(), 'big') / 2**64


is_val = np.array([hsh(find(a)) < 0.2 for a in h1])
log(f'human: train {(~is_val).sum()}, val {is_val.sum()}')

# ---------------------------------------------------------------- тексты
need_llm = (set(llm_train['id1'].to_list()) | set(llm_train['id2'].to_list()) | hold_items)
log(f'подгружаю {len(need_llm)} товаров из полного каталога...')
t = time.time()
cat_items = (pl.scan_parquet(f'{DATA}/items.parquet')
             .filter(pl.col('id').is_in(list(need_llm)))
             .select('id', 'name', 'attributes', 'category')
             .collect(engine='streaming'))
log(f'подгружено {cat_items.height} за {time.time() - t:.0f} c')

TEXT, CAT = {}, {}
for i, n, a, c in zip(cat_items['id'].to_list(), cat_items['name'].to_list(),
                      cat_items['attributes'].to_list(), cat_items['category'].to_list()):
    TEXT[i] = preprocess_item(n, a, c)['text']
    CAT[i] = c
del cat_items

hum_items = pl.read_parquet(f'{DATA}/items_human.parquet')
for i, n, a, c in zip(hum_items['id'].to_list(), hum_items['name'].to_list(),
                      hum_items['attributes'].to_list(), hum_items['category'].to_list()):
    TEXT[i] = preprocess_item(n, a, c)['text']
    CAT[i] = c
del hum_items
log(f'текстов готово: {len(TEXT)}')


class Pairs(Dataset):
    def __init__(self, a, b, y, swap):
        self.a, self.b, self.y, self.swap = a, b, y, swap

    def __len__(self):
        return len(self.a)

    def __getitem__(self, k):
        x, z = TEXT[self.a[k]], TEXT[self.b[k]]
        if self.swap and np.random.rand() < 0.5:
            x, z = z, x
        return x, z, self.y[k]


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


def collate(batch):
    a, b, y = zip(*batch)
    enc = tokenizer(list(a), list(b), padding=True, truncation=True,
                    max_length=MAX_LENGTH, return_tensors='pt')
    enc['labels'] = torch.tensor(y, dtype=torch.float32)
    return enc


def loader(ds, shuffle, bs=BATCH):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate,
                      num_workers=6, pin_memory=True, drop_last=shuffle)


llm_dl = loader(Pairs(llm_train['id1'].to_list(), llm_train['id2'].to_list(),
                      llm_train['target'].to_numpy().astype(np.float32), True), True)
hum_tr = np.flatnonzero(~is_val)
hum_dl = loader(Pairs([h1[i] for i in hum_tr], [h2[i] for i in hum_tr], hy[hum_tr], True), True)

hum_va = np.flatnonzero(is_val)
hum_val_dl = loader(Pairs([h1[i] for i in hum_va], [h2[i] for i in hum_va], hy[hum_va], False), False, BATCH * 2)
hum_val_y, hum_val_cat = hy[hum_va], np.array([CAT[h1[i]] for i in hum_va])

hold_dl = loader(Pairs(holdout['id1'].to_list(), holdout['id2'].to_list(),
                       holdout['target'].to_numpy().astype(np.float32), False), False, BATCH * 2)
hold_y = holdout['target'].to_numpy()
hold_cat = np.array([CAT[i] for i in holdout['id1'].to_list()])

# ---------------------------------------------------------------- модель
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=1, dtype=torch.float32).cuda()
loss_fn = torch.nn.BCEWithLogitsLoss()
log(f'модель {BASE_MODEL} загружена')


def macro(y_true, score, groups, mn=50):
    per = [average_precision_score(y_true[groups == g], score[groups == g])
           for g in np.unique(groups)
           if (groups == g).sum() >= mn and 0 < y_true[groups == g].sum() < (groups == g).sum()]
    return float(np.mean(per))


@torch.inference_mode()
def scores_for(dl):
    model.eval()
    out = []
    for batch in dl:
        batch.pop('labels')
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out.append(model(**batch).logits.squeeze(-1).float().cpu().numpy())
    model.train()
    return np.concatenate(out)


def evaluate(tag, step):
    hv = macro(hum_val_y, scores_for(hum_val_dl), hum_val_cat)
    hd = macro(hold_y, scores_for(hold_dl), hold_cat)
    log(f'  ОЦЕНКА [{tag} шаг {step}] human_val={hv:.4f} (опора 0.7750) | '
        f'llm_holdout={hd:.4f} (опора 0.5927)')
    return hv, hd


def train(dl, epochs, lr, tag, eval_every):
    steps = int(len(dl) * epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    log(f'=== {tag}: {steps} шагов, batch={BATCH}, lr={lr} ===')
    model.train()
    step, run, t0, best = 0, 0.0, time.time(), -1.0
    done = False
    while not done:
        for batch in dl:
            labels = batch.pop('labels').cuda(non_blocking=True)
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = loss_fn(model(**batch).logits.squeeze(-1).float(), labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
            step += 1
            if step % 200 == 0:
                r = step / (time.time() - t0)
                log(f'  {tag} {step}/{steps} loss={run/200:.4f} ({r:.2f} it/s, '
                    f'осталось {(steps-step)/r/60:.0f} мин)')
                run = 0.0
            if step % eval_every == 0 or step == steps:
                hv, hd = evaluate(tag, step)
                if hd > best:
                    best = hd
                    model.save_pretrained(f'{OUT}/best_{tag}')
                    tokenizer.save_pretrained(f'{OUT}/best_{tag}')
                    json.dump({'stage': tag, 'step': step, 'human_val': hv, 'llm_holdout': hd},
                              open(f'{OUT}/metrics_{tag}.json', 'w'), ensure_ascii=False, indent=2)
                    log(f'  сохранено (лучший llm_holdout={best:.4f})')
            if step >= steps:
                done = True
                break
    return best


log('исходная модель до обучения:')
evaluate('стадия0', 0)

train(llm_dl, 1.0, LR_STAGE1, 'llm', 3000)
train(hum_dl, 1.0, LR_STAGE2, 'human', 1500)
log('ГОТОВО')
