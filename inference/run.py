"""Точка входа решения.

    python -u run.py --items_path items.parquet --matches_path matches.parquet \
                     --output-path submit.csv

Выход: csv с колонками id1, id2, predict — строка на каждую входную пару.

Зависимости намеренно минимальны (pyarrow + torch + transformers): образ для
запуска задан организаторами, а интернета в рантайме нет.
"""
import argparse
import csv
import os
import sys
import time

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.inference import FALLBACK_SCORE, CrossEncoder, log
from src.preprocess import preprocess_item

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'cross_encoder')


def parse_args():
    p = argparse.ArgumentParser()
    # в спецификации output пишется через дефис, а items/matches через
    # подчёркивание — принимаем оба варианта, чтобы не упасть на запуске
    p.add_argument('--items_path', '--items-path', '-i', required=True)
    p.add_argument('--matches_path', '--matches-path', '-m', required=True)
    p.add_argument('--output-path', '--output_path', '-o', required=True, dest='output_path')
    p.add_argument('--batch_size', type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    pairs = pq.read_table(args.matches_path, columns=['id1', 'id2'])
    id1 = pairs.column('id1').to_pylist()
    id2 = pairs.column('id2').to_pylist()
    log(f'пар на вход: {len(id1)}')

    # Готовим только те товары, что реально встречаются в парах: тестовый
    # каталог может быть заметно больше выборки пар.
    needed = set(id1) | set(id2)
    log(f'нужно товаров: {len(needed)}')

    dataset = ds.dataset(args.items_path, format='parquet')
    table = dataset.to_table(
        columns=['id', 'name', 'attributes', 'category'],
        filter=pc.field('id').isin(list(needed)),
    )
    log(f'найдено товаров в каталоге: {table.num_rows}')

    texts = {}
    for item_id, name, attributes, category in zip(
            table.column('id').to_pylist(), table.column('name').to_pylist(),
            table.column('attributes').to_pylist(), table.column('category').to_pylist()):
        texts[item_id] = preprocess_item(name, attributes, category)['text']
    log(f'тексты готовы, {time.time() - t0:.0f} c от старта')

    missing = len(needed) - len(texts)
    if missing:
        log(f'ВНИМАНИЕ: {missing} товаров нет в каталоге, их пары получат {FALLBACK_SCORE}')

    # Пары с двумя известными товарами считаем моделью, остальным ставим
    # fallback. В выдачу попадают ВСЕ пары без исключения — этого требуют правила.
    known = [k for k in range(len(id1)) if id1[k] in texts and id2[k] in texts]
    left = [texts[id1[k]] for k in known]
    right = [texts[id2[k]] for k in known]

    model = CrossEncoder(MODEL_DIR, batch_size=args.batch_size)
    scores = model.predict(left, right)

    predict = [FALLBACK_SCORE] * len(id1)
    for k, s in zip(known, scores):
        predict[k] = float(s)

    with open(args.output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id1', 'id2', 'predict'])
        writer.writerows(zip(id1, id2, predict))

    log(f'записано {len(id1)} строк в {args.output_path}, всего {(time.time() - t0) / 60:.1f} мин')


if __name__ == '__main__':
    main()
