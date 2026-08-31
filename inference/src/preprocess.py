"""Предобработка товаров для матчинга через deepvk/USER-bge-m3.

Единый источник правды: и офлайн-чистка каталога, и инференс на одном товаре
идут через одну и ту же функцию `preprocess_item`, чтобы тексты, на которых
строились эмбеддинги, не разъехались с тем, что подаётся в модель в рантайме.

    from scripts.preprocess import preprocess_item, preprocess_frame

    # один товар (инференс)
    item = preprocess_item(name='...', attributes='{"бренд":"..."}', category='Обувь')
    text = item['text']

    # весь каталог (офлайн)
    clean = preprocess_frame(pl.scan_parquet('data/items.parquet'))
"""

from __future__ import annotations

import html
import json
import re


# --- Нормализация ключей атрибутов -------------------------------------------
# В каталоге 72k уникальных ключей, и одно поле часто названо по-разному у
# разных поставщиков. Склеиваем самые массовые синонимы, иначе один и тот же
# товар от двух поставщиков даёт разный текст.
KEY_SYNONYMS = {
    "страна производства": "страна",
    "страна-изготовитель": "страна",
    "страна-производитель": "страна",
    "страна производитель": "страна",
    "цвет товара": "цвет",
    "название цвета": "цвет",
    "партномер (артикул производителя)": "артикул",
    "артикул производителя": "артикул",
    "номер артикула": "артикул",
    "модель, артикул производителя": "модель",
    "бренд в одежде и обуви": "бренд",
    "российский размер": "размер",
    "размер производителя": "размер",
}

# --- Приоритет источника при коллизии синонимов --------------------------------
# У 16.8% товаров каталога канонический ключ собирается из двух разных исходных
# полей, и у 9.9% их значения РАЗЛИЧАЮТСЯ. Раньше побеждало то, что стояло раньше
# в JSON, — то есть выбор был случайным.
#
# Порядок измерен напрямую: на положительных парах взяты товары с коллизией и
# посчитано, значение из какого источника чаще совпадает со значением партнёра.
# Стороны пары приходят из разных схем атрибутов, поэтому это единственный
# способ сравнить источники — обычный lift тут не считается (пар, где один и тот
# же исходный ключ есть у обоих, почти нет).
SOURCE_PRIORITY = {
    # 30.3% / 13.9% / 11.4% совпадений с партнёром
    "цвет": ("цвет товара", "название цвета", "цвет"),
    # 58.2% у партномера производителя против 0.9% у артикула маркетплейса
    "артикул": ("партномер (артикул производителя)", "артикул производителя",
                "номер артикула", "артикул"),
    # 6.3% / 5.6% / 0.4%
    "размер": ("размер производителя", "российский размер", "размер"),
}

# (канонический ключ, исходный ключ) -> ранг; меньше — предпочтительнее
_SOURCE_RANK = {(canon, src): i
                for canon, order in SOURCE_PRIORITY.items()
                for i, src in enumerate(order)}
_NO_RANK = 99


# Порядок атрибутов в тексте фиксирован: одинаковый товар всегда даёт
# одинаковую строку. Сначала самые различающие поля.
PRIORITY_KEYS = (
    "бренд",
    "модель",
    "артикул",
    "тип",
    "цвет",
    "размер",
    "материал",
    "состав",
    "объём",
    "вес товара, г",
    "вид техники",
    "пол",
)

# Поля, которые не помогают отличить один товар от другого: маркетинговый шум,
# логистика, одинаковые у всей категории.
BLOCKED_KEYS = frozenset(
    {
        "#хештеги",
        "хештеги",
        "гарантийный срок",
        "срок гарантии",
        "валюта",
        "ширина упаковки",
        "длина упаковки",
        "высота упаковки",
        "вес товара с упаковкой (г)",
        "вес с упаковкой (кг)",
        "размер упаковки (длина х ширина х высота), см",
        "штрих-код",
        "тнвэд",
        "ключевые слова",
    }
)

# Значения, означающие "данных нет" — в тексте от них только шум.
# "да"/"нет"/"0" оставляем: это настоящие значения признаков.
NULL_VALUES = frozenset(
    {"", "-", "—", "--", "не указано", "нет данных", "не указан", "n/a", "na", "none", "null"}
)

_HTML_TAG = re.compile(r"<[a-zA-Z/][^>]*>")
_WHITESPACE = re.compile(r"\s+")
# Управляющие символы и «невидимки», которые ломают токенизацию.
_INVISIBLE = re.compile(r"[\xad\u200b-\u200f\u2028\u2029\u202f\ufeff\x00-\x1f]")

MAX_NAME_LEN = 300
MAX_VALUE_LEN = 100
MAX_ATTRS = 16


def clean_text(value: object) -> str:
    """Привести текстовое поле к нормальному виду.

    Снимает html, разворачивает entities, убирает невидимые символы и
    неразрывные пробелы, схлопывает пробелы, приводит к нижнему регистру.
    """
    if value is None:
        return ""
    text = str(value)
    if "&" in text:
        text = html.unescape(text)
    if "<" in text:
        text = _HTML_TAG.sub(" ", text)
    text = text.replace("\xa0", " ")
    text = _INVISIBLE.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip().lower()


def normalize_attributes(raw: object) -> dict[str, str]:
    """Разобрать attributes и привести к канону: ключи-синонимы склеены,
    шумные поля и пустые значения выброшены, значения обрезаны."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return {}
    elif isinstance(raw, dict):
        parsed = raw
    else:
        return {}

    result: dict[str, str] = {}
    ranks: dict[str, int] = {}
    for key, value in parsed.items():
        source = clean_text(key)
        if not source or source in BLOCKED_KEYS:
            continue
        key = KEY_SYNONYMS.get(source, source)
        if key in BLOCKED_KEYS:
            continue

        value = clean_text(value)
        if value in NULL_VALUES:
            continue
        if len(value) > MAX_VALUE_LEN:
            value = value[:MAX_VALUE_LEN].rstrip()
        # Побеждает значение из более приоритетного источника, а не то, что
        # раньше встретилось в JSON. Для ключей вне SOURCE_PRIORITY ранг у всех
        # одинаковый, и поведение совпадает с прежним setdefault.
        rank = _SOURCE_RANK.get((key, source), _NO_RANK)
        if key not in result or rank < ranks[key]:
            result[key] = value
            ranks[key] = rank
    return result


def build_item_text(name: str, attributes: dict[str, str], category: str = "") -> str:
    """Собрать строку, которая пойдёт в USER-bge-m3.

    Порядок детерминирован: категория, название, затем приоритетные атрибуты,
    затем остальные по алфавиту.
    """
    if len(name) > MAX_NAME_LEN:
        name = name[:MAX_NAME_LEN].rstrip()

    parts = []
    if category:
        parts.append(category)
    if name:
        parts.append(name)

    used = set()
    for key in PRIORITY_KEYS:
        value = attributes.get(key)
        if value:
            parts.append(f"{key}: {value}")
            used.add(key)

    remaining = MAX_ATTRS - len(used)
    if remaining > 0:
        for key in sorted(attributes):
            if key in used:
                continue
            parts.append(f"{key}: {attributes[key]}")
            remaining -= 1
            if remaining == 0:
                break

    return " | ".join(parts)


def preprocess_item(name: object, attributes: object, category: object = "") -> dict:
    """Предобработать один товар. Используется и в чистке, и в инференсе."""
    clean_name = clean_text(name)
    clean_category = clean_text(category)
    attrs = normalize_attributes(attributes)
    return {
        "name": clean_name,
        "attributes": attrs,
        "category": clean_category,
        "text": build_item_text(clean_name, attrs, clean_category),
    }


def preprocess_frame(
    source,
    batch_size: int = 200_000,
    progress: bool = False,
) -> pl.DataFrame:
    """Предобработать таблицу товаров, построчно через `preprocess_item`.

    `source` — путь к parquet, LazyFrame или DataFrame. polars импортируется
    лениво: инференсу нужен только `preprocess_item`, без тяжёлых зависимостей. Parquet читается
    батчами через pyarrow, чтобы 13.4M строк не держать в памяти целиком.
    Возвращает id, name, category, text и n_attrs.
    """
    import polars as pl
    import pyarrow.parquet as pq

    columns = ["id", "name", "attributes", "category"]

    if isinstance(source, str):
        batches = (
            {c: b.column(c).to_pylist() for c in columns}
            for b in pq.ParquetFile(source).iter_batches(batch_size=batch_size, columns=columns)
        )
    else:
        frame = source.collect() if isinstance(source, pl.LazyFrame) else source
        batches = (
            frame.slice(offset, batch_size).select(columns).to_dict(as_series=False)
            for offset in range(0, frame.height, batch_size)
        )

    chunks = []
    done = 0
    for batch in batches:
        rows = []
        for item_id, name, attributes, category in zip(
            batch["id"], batch["name"], batch["attributes"], batch["category"]
        ):
            item = preprocess_item(name, attributes, category)
            rows.append(
                {
                    "id": item_id,
                    "name": item["name"],
                    "category": item["category"],
                    "text": item["text"],
                    "n_attrs": len(item["attributes"]),
                }
            )
        chunks.append(pl.DataFrame(rows))
        done += len(rows)
        if progress:
            print(f"  обработано {done}", flush=True)

    if not chunks:
        return pl.DataFrame(schema={"id": pl.Int64, "name": pl.Utf8, "category": pl.Utf8,
                                    "text": pl.Utf8, "n_attrs": pl.Int64})
    return pl.concat(chunks)


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Чистка каталога товаров")
    parser.add_argument("--input", default="data/items_human.parquet")
    parser.add_argument("--output", default="data/items_human_clean.parquet")
    args = parser.parse_args()

    start = time.time()
    clean = preprocess_frame(args.input, progress=True)
    clean.write_parquet(args.output)
    print(f"{clean.height} товаров -> {args.output} за {time.time() - start:.1f} c")
