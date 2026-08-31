"""Сквозная проверка: даёт ли упакованное решение ту же метрику, что и обучение.

Гоняем run.py на val-парах и считаем macro PR-AUC по выходному csv.
Расхождение с метрикой обучения означало бы рассинхрон инференса и обучения.
"""
import csv
import hashlib
import subprocess
import time

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

DATA = '/home/jovyan/xakaton/data'
SEED = 42

matches = pl.read_parquet(f'{DATA}/matches.parquet')
id1, id2 = matches['id1'].to_list(), matches['id2'].to_list()
y = matches['target'].to_numpy()

parent = {}


def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for a, b in zip(id1, id2):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb


def h(root, seed=SEED):
    return int.from_bytes(hashlib.blake2b(f'{seed}:{root}'.encode(), digest_size=8).digest(), 'big') / 2**64


is_val = np.array([h(find(a)) < 0.2 for a in id1])
vidx = np.flatnonzero(is_val)
print(f'val пар: {len(vidx)}')

val_pairs = matches[vidx.tolist()].select('id1', 'id2')
val_pairs.write_parquet('/tmp/val_pairs.parquet')

t = time.time()
r = subprocess.run(
    ['/home/jovyan/xakaton/venv/bin/python', '-u', 'run.py',
     '--items_path', f'{DATA}/items_human.parquet',
     '--matches_path', '/tmp/val_pairs.parquet',
     '--output-path', '/tmp/submit_val.csv'],
    cwd='/home/jovyan/xakaton/solution', capture_output=True, text=True, timeout=3600)
print(r.stdout[-1200:])
if r.returncode:
    print('ОШИБКА', r.stderr[-1500:])
    raise SystemExit(1)

pred = {}
with open('/tmp/submit_val.csv') as f:
    for row in csv.DictReader(f):
        pred[(int(row['id1']), int(row['id2']))] = float(row['predict'])

scores = np.array([pred[(id1[i], id2[i])] for i in vidx])
vy = y[vidx]
items = pl.read_parquet(f'{DATA}/items_human.parquet').select('id', 'category')
category = dict(zip(items['id'].to_list(), items['category'].to_list()))
vcat = np.array([category[id1[i]] for i in vidx])


def macro(y_true, score, groups, min_size=50):
    per = [average_precision_score(y_true[groups == g], score[groups == g])
           for g in np.unique(groups)
           if (groups == g).sum() >= min_size and 0 < y_true[groups == g].sum() < (groups == g).sum()]
    return float(np.mean(per))


m = macro(vy, scores, vcat)
print(f'\nMACRO PR-AUC упакованного решения: {m:.4f}')
print(f'метрика на обучении (эпоха 1):     0.7750')
print(f'расхождение: {abs(m - 0.7750):.4f}')
print(f'скорость: {len(vidx) / (time.time() - t):.0f} пар/сек -> '
      f'~{390_000 / (len(vidx) / (time.time() - t)) / 60:.0f} мин на 390k тестовых пар')
