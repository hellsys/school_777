# school_777

Материалы к учебному кейсу «Человек или подделка?» (Ozon Банк, датасет CelebA-Spoof).

## celeba_spoof_loader.py

Скачивает часть датасета CelebA-Spoof: по `n_per_class` настоящих лиц и подделок, рамки лиц `_BB.txt`
и разметку `metas/intra_test/train_label.json`. На 1000 картинок каждого класса уходит 3–5 минут и около 200 МБ.

В ноутбуке (Google Colab или локально) вместо технической ячейки загрузки:

```python
!wget -q -O celeba_spoof_loader.py https://raw.githubusercontent.com/hellsys/school_777/main/celeba_spoof_loader.py
from celeba_spoof_loader import download_celeba_spoof
download_celeba_spoof(DATA_DIR, N_PER_CLASS, seed=SEED)
```

Из командной строки:

```bash
python celeba_spoof_loader.py --dir CelebA_Spoof --n 1000
```

Повторный запуск ничего не качает заново. Если увеличить `N_PER_CLASS`, докачает недостающее.
Нужна библиотека `requests` (в Colab уже есть).

> CelebA-Spoof можно использовать только для учёбы и исследований. Не публикуйте картинки и не выкладывайте датасет в открытый доступ.
