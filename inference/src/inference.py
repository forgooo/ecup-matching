"""Инференс cross-encoder'а для матчинга товаров.

Работает без интернета: модель грузится строго из локальной папки.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LENGTH = 384
# скор для пар, где товара нет в каталоге: нейтральный, но пара всё равно
# попадает в выдачу — правила требуют результат для КАЖДОЙ пары
FALLBACK_SCORE = 0.0


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


class CrossEncoder:
    def __init__(self, model_dir: str, batch_size: int = 128):
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if self.device == 'cuda' else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=dtype)
        self.model = self.model.to(self.device).eval()
        log(f'модель загружена: {model_dir}, device={self.device}, dtype={dtype}')

    @torch.inference_mode()
    def predict(self, left: list[str], right: list[str]) -> np.ndarray:
        """Вероятности матча. Порядок выхода совпадает с порядком входа."""
        n = len(left)
        if n == 0:
            return np.zeros(0, dtype=np.float32)

        # сортировка по длине: в батче тексты схожего размера, паддинга почти нет
        order = np.argsort([len(a) + len(b) for a, b in zip(left, right)])
        out = np.empty(n, dtype=np.float32)

        start = time.time()
        for i in range(0, n, self.batch_size):
            idx = order[i:i + self.batch_size]
            enc = self.tokenizer([left[j] for j in idx], [right[j] for j in idx],
                                 padding=True, truncation=True,
                                 max_length=MAX_LENGTH, return_tensors='pt')
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits.squeeze(-1).float()
            out[idx] = torch.sigmoid(logits).cpu().numpy()

            done = i + len(idx)
            if done % (self.batch_size * 200) == 0:
                rate = done / (time.time() - start)
                log(f'  {done}/{n} ({rate:.0f} пар/сек, осталось {(n - done) / rate / 60:.1f} мин)')

        log(f'предсказано {n} пар за {(time.time() - start) / 60:.1f} мин')
        return out
