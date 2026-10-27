"""Загрузчик части датасета CelebA-Spoof для учебного ноутбука.

Использование в ноутбуке (Google Colab или локально):

    !wget -q -O celeba_spoof_loader.py https://raw.githubusercontent.com/hellsys/school_777/main/celeba_spoof_loader.py
    from celeba_spoof_loader import download_celeba_spoof
    download_celeba_spoof(DATA_DIR, N_PER_CLASS, seed=SEED)

Или из командной строки:

    python celeba_spoof_loader.py --dir CelebA_Spoof --n 1000

Как это работает: весь датасет — это один огромный zip-архив (77 ГБ), разрезанный на 8 кусков.
В конце любого zip-архива лежит «оглавление»: какой файл где лежит внутри архива.
Мы скачиваем только оглавление и разметку, выбираем нужное число картинок
и скачиваем из архива только их, «вырезая» нужные байты.

Повторный запуск ничего не качает заново. Если увеличить число картинок, докачает недостающее.
Источник: копия архива CelebA-Spoof на Hugging Face (namkuner/CelebA-Spoof-Dataset).
"""

import json
import random
import struct
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import requests

ARCHIVE_URL = "https://huggingface.co/datasets/namkuner/CelebA-Spoof-Dataset/resolve/main/data/part_{:02d}.zip"
ARCHIVE_PARTS = 8
ARCHIVE_PREFIX = "CelebA_Spoof_/CelebA_Spoof/"
LABELS_NAME = "metas/intra_test/train_label.json"

_thread_data = threading.local()


def _session():
    # У каждого потока своё соединение с сервером
    if not hasattr(_thread_data, "session"):
        _thread_data.session = requests.Session()
    return _thread_data.session


class RemoteArchive:
    def __init__(self):
        self.urls = [None] * ARCHIVE_PARTS
        self.sizes = [0] * ARCHIVE_PARTS
        for part in range(ARCHIVE_PARTS):
            self.resolve(part)
        self.starts = [sum(self.sizes[:i]) for i in range(ARCHIVE_PARTS)]
        self.total_size = sum(self.sizes)

    def resolve(self, part):
        # Hugging Face перенаправляет на файловый сервер. Узнаём прямую ссылку один раз:
        # у Hugging Face есть лимит запросов (3000 за 5 минут на один IP), а у файлового сервера его нет.
        for attempt in range(10):
            response = requests.head(ARCHIVE_URL.format(part), allow_redirects=True, timeout=60)
            if response.status_code == 429:
                wait = int(response.headers.get("Retry-After", 30))
                print(f"  Hugging Face просит подождать {wait} с...")
                time.sleep(wait + 1)
                continue
            response.raise_for_status()
            self.urls[part] = response.url
            self.sizes[part] = int(response.headers["Content-Length"])
            return
        raise RuntimeError("Hugging Face не отвечает, попробуйте запустить ячейку позже")

    def read(self, offset, length):
        # Читаем байты [offset, offset + length) так, будто все куски склеены в один файл
        result = b""
        while length > 0:
            part = max(i for i in range(ARCHIVE_PARTS) if self.starts[i] <= offset)
            local = offset - self.starts[part]
            n = min(length, self.sizes[part] - local)
            for attempt in range(6):
                try:
                    response = _session().get(
                        self.urls[part], headers={"Range": f"bytes={local}-{local + n - 1}"}, timeout=60
                    )
                    if response.status_code == 206 and len(response.content) == n:
                        break
                    if response.status_code == 403:  # прямая ссылка устарела, получаем новую
                        self.resolve(part)
                except requests.RequestException:
                    pass
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError("Не получилось скачать кусок архива")
            result += response.content
            offset += n
            length -= n
        return result

    def read_directory(self):
        # Оглавление zip64: в конце архива лежит указатель на него
        tail = self.read(self.total_size - 65536, 65536)
        locator = tail.rfind(b"PK\x06\x07")
        record_offset = struct.unpack("<Q", tail[locator + 8:locator + 16])[0]
        record = self.read(record_offset, 56)
        directory_size, directory_offset = struct.unpack("<QQ", record[40:56])
        return self.read(directory_offset, directory_size)

    def extract(self, entry):
        offset, compressed_size, method = entry
        # Читаем заголовок файла и сами данные одним запросом (с запасом на длину заголовка)
        chunk = self.read(offset, min(30 + 512 + compressed_size, self.total_size - offset))
        name_length, extra_length = struct.unpack("<HH", chunk[26:30])
        start = 30 + name_length + extra_length
        data = chunk[start:start + compressed_size]
        if len(data) < compressed_size:
            data = self.read(offset + start, compressed_size)
        return data if method == 0 else zlib.decompress(data, -15)


def find_in_directory(directory, wanted_names):
    # Ищем в оглавлении нужные файлы: имя -> (где лежит, сколько байт, способ сжатия)
    found = {}
    pos = 0
    while pos < len(directory):
        method = struct.unpack("<H", directory[pos + 10:pos + 12])[0]
        compressed_size, size = struct.unpack("<II", directory[pos + 20:pos + 28])
        name_length, extra_length, comment_length = struct.unpack("<HHH", directory[pos + 28:pos + 34])
        offset = struct.unpack("<I", directory[pos + 42:pos + 46])[0]
        name = directory[pos + 46:pos + 46 + name_length].decode("utf-8", "replace")
        if name in wanted_names:
            extra = directory[pos + 46 + name_length:pos + 46 + name_length + extra_length]
            e = 0
            while e < len(extra):
                header_id, data_length = struct.unpack("<HH", extra[e:e + 4])
                if header_id == 1:  # zip64: настоящие размеры и смещение лежат здесь
                    values = extra[e + 4:e + 4 + data_length]
                    v = 0
                    if size == 0xFFFFFFFF:
                        v += 8
                    if compressed_size == 0xFFFFFFFF:
                        compressed_size = struct.unpack("<Q", values[v:v + 8])[0]
                        v += 8
                    if offset == 0xFFFFFFFF:
                        offset = struct.unpack("<Q", values[v:v + 8])[0]
                e += 4 + data_length
            found[name] = (offset, compressed_size, method)
        pos += 46 + name_length + extra_length + comment_length
    return found


def count_local_images(data_dir):
    counts = {"live": 0, "spoof": 0}
    for label in counts:
        counts[label] = sum(1 for p in data_dir.glob(f"Data/train/*/{label}/*") if not p.name.endswith("_BB.txt"))
    return counts


def download_celeba_spoof(data_dir, n_per_class, seed=42):
    counts = count_local_images(data_dir)
    if (data_dir / LABELS_NAME).exists() and min(counts.values()) >= n_per_class:
        print("Данные уже скачаны:", counts)
        return

    print("1/4 Читаем оглавление архива (~150 МБ)...")
    archive = RemoteArchive()
    directory = archive.read_directory()

    print("2/4 Скачиваем разметку...")
    entry = find_in_directory(directory, {ARCHIVE_PREFIX + LABELS_NAME})[ARCHIVE_PREFIX + LABELS_NAME]
    labels = json.loads(archive.extract(entry))

    # Случайно выбираем по n_per_class картинок каждого класса (одинаково при одинаковом seed)
    rng = random.Random(seed)
    chosen = []
    for label in ["live", "spoof"]:
        keys = sorted(k for k in labels if f"/{label}/" in k)
        rng.shuffle(keys)
        chosen += keys[:n_per_class]

    wanted = []
    for key in chosen:
        wanted.append(key)
        wanted.append(key.rsplit(".", 1)[0] + "_BB.txt")
    missing = [name for name in wanted if not (data_dir / name).exists()]
    entries = find_in_directory(directory, {ARCHIVE_PREFIX + name for name in missing})
    missing = [name for name in missing if ARCHIVE_PREFIX + name in entries]
    del directory

    print("3/4 Сохраняем разметку...")
    label_path = data_dir / LABELS_NAME
    label_path.parent.mkdir(parents=True, exist_ok=True)
    saved_labels = json.loads(label_path.read_text()) if label_path.exists() else {}
    saved_labels.update({key: labels[key] for key in chosen})
    label_path.write_text(json.dumps(saved_labels))

    print(f"4/4 Скачиваем {len(missing)} файлов (картинки и рамки лиц)...")

    def save(name):
        try:
            data = archive.extract(entries[ARCHIVE_PREFIX + name])
        except Exception:
            return False
        path = data_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return True

    failed = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, ok in enumerate(pool.map(save, missing)):
            failed += not ok
            if (i + 1) % 500 == 0 or i + 1 == len(missing):
                print(f"  скачано {i + 1} из {len(missing)}")

    print("Готово!", count_local_images(data_dir))
    if failed:
        print(f"Не удалось скачать {failed} файлов. Запустите ячейку ещё раз: она докачает только недостающее.")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Скачать часть датасета CelebA-Spoof")
    parser.add_argument("--dir", default="CelebA_Spoof", help="папка для данных (по умолчанию CelebA_Spoof)")
    parser.add_argument("--n", type=int, default=1000, help="сколько картинок каждого класса (по умолчанию 1000)")
    parser.add_argument("--seed", type=int, default=42, help="зерно случайности (по умолчанию 42)")
    args = parser.parse_args()
    download_celeba_spoof(Path(args.dir), args.n, seed=args.seed)
