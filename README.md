# ДЗ 2.4 + 2.5

Репозиторий содержит два эксперимента:

- **HW 2.4:** fine-tuning DETR на COCO-subset (≥10 классов) + TensorBoard + profiler trace + графики лоссов + error analysis.
- **HW 2.5:** аугментация «редких» классов синтетикой (Stable Diffusion + ControlNet) и ablation real vs real+synth на задаче классификации кропов.

---

## Структура репозитория

- `src/detr/` — подготовка COCO/subset, обучение DETR, оценка, визуализации, error analysis  
- `src/synth/` — выбор редких классов, создание датасета кропов, генерация синтетики, обучение классификатора, ablation-таблица  
- `runs/` — артефакты экспериментов (TensorBoard, checkpoints, метрики, графики, анализ ошибок, ablation)  
- `examples_synth/` — небольшая выборка синтетических изображений для README (не весь датасет)
- `tb/` - TensorBoard
- `plots/loss_curves.png` - Loss plots
- `metrics/val_metrics.json` - COCO metrics
- `viz/` - Визуализации
- `analysis/` - Error analysis 


---

# Задание 2.4 

## 1) Датасет (COCO-subset, 10 классов)

Источник: **MS COCO 2017** (train/val изображения + `instances_train2017.json`, `instances_val2017.json`).  
Subset собран по 10 категориям:

- `person`
- `bicycle`
- `car`
- `motorcycle`
- `bus`
- `train`
- `truck`
- `boat`
- `traffic light`
- `fire hydrant`
---

## 2) Модель
- Архитектура: **DETR ResNet-50**
- Pretrained: `facebook/detr-resnet-50`
- Fine-tuning на `K=10` классов (`ignore_mismatched_sizes=True`)

---

## 3) Гиперпараметры обучения
- epochs: **10**
- batch_size: **2**
- optimizer: **AdamW**
- lr: **1e-4**
- lr_backbone: **1e-5**
- weight_decay: **1e-4**
- AMP: **on**

Артефакты:
- TensorBoard: `tb/`
- Loss plots: `plots/loss_curves.png`
- COCO metrics: `metrics/val_metrics.json`
- Визуализации: `viz/`
- Error analysis: `analysis/`

---

## 4) Запуск 

### Скачать COCO 2017
```bash
python -m src.detr.prepare_coco --data_root data/coco --download
```

### Собрать subset (10 классов)
```bash
python -m src.detr.prepare_coco_subset \
  --coco_root data/coco \
  --out_root data/coco_subset \
  --classes person bicycle car motorcycle bus train truck boat "traffic light" "fire hydrant"
```

### Симлинки на изображения (чтобы не копировать гигабайты)
```bash
mkdir -p data/coco_subset/images
ln -s ../../coco/images/train2017 data/coco_subset/images/train2017
ln -s ../../coco/images/val2017   data/coco_subset/images/val2017
```

### Обучение DETR
```bash
python -m src.detr.train \
  --data_root data/coco_subset \
  --model_name facebook/detr-resnet-50 \
  --output_dir runs/detr_coco_subset \
  --epochs 10 \
  --batch_size 2 \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --weight_decay 1e-4 \
  --amp
```

### TensorBoard + profiler trace
```bash
tensorboard --logdir runs/detr_coco_subset/tb
```
## 5) Метрики (COCO mAP/mAP50)

- Для COCOeval: используем score_thr=0.0 (без отсечения).
- Для визуализаций/анализа: можно score_thr=0.2–0.3 для более “чистых” картинок.

DETR на ранних эпохах может выдавать низкие confidence scores. Порог (например 0.3) отсекает часть корректных детекций, снижая recall и mAP. Для COCOeval корректнее не отсекать предсказания и оценивать ранжирование по score.

### Оценка метрик:

```bash
python -m src.detr.eval_coco \
  --data_root data/coco_subset \
  --checkpoint runs/detr_coco_subset/checkpoints/best.pt \
  --model_name facebook/detr-resnet-50 \
  --output_dir runs/detr_coco_subset \
  --batch_size 2 \
  --score_thr 0.0
```

| split | mAP | mAP50 |
|---|---:|---:|
| val | 0.1086562972347361 | 0.23755006134598128 |

## 6) Графики потерь (classification + bbox regression)

В DETR основные компоненты:
- Classification loss: loss_ce
- BBox regression losses: loss_bbox (L1) и loss_giou (GIoU)
- Total: loss_total

График:
plots/loss_curves.png

TensorBoard-скаляры:
train/loss_ce, train/loss_bbox, train/loss_giou, train/loss_total

## 7) Визуализации боксов (qualitative)

```bash
python -m src.detr.visualize \
  --data_root data/coco_subset \
  --checkpoint runs/detr_coco_subset/checkpoints/best.pt \
  --model_name facebook/detr-resnet-50 \
  --output_dir runs/detr_coco_subset \
  --num_images 30 \
  --score_thr 0.3
```

Результат находится в папке viz

## 8) Error analysis (classification vs localization)

```bash
python -m src.detr.error_analysis \
  --data_root data/coco_subset \
  --checkpoint runs/detr_coco_subset/checkpoints/best.pt \
  --model_name facebook/detr-resnet-50 \
  --output_dir runs/detr_coco_subset \
  --iou_thr 0.5 \
  --score_thr 0.3 \
  --max_images 400
```

Результаты находятся в папке analysis
Интерпретация результатов:
- cls_error: IoU ≥ 0.5, но неверный класс
- loc_error: IoU < 0.5 (плохая локализация)
- missed: объект не найден
- false_pos: лишний бокс без GT-сопадения

Разрыв между mAP50 (0.238) и mAP@[0.50:0.95] (0.109) указывает, что модель часто “примерно” находит объекты, но точность локализации на строгих IoU-порогах ещё ограничена.
По размерам объектов качество выше на крупных объектах и хуже на small — типичное поведение для DETR при коротком fine-tuning.

| split | mAP@[0.50:0.95] | mAP50 | AP75 |
|---|---:|---:|---:|
| val | 0.109 | 0.238 | 0.091 |

### По размерам объектов (AP@[0.50:0.95])
| area | AP |
|---|---:|
| small | 0.032 |
| medium | 0.097 |
| large | 0.191 |

### Recall (AR@[0.50:0.95])
| maxDets | AR (all) | AR (small) | AR (medium) | AR (large) |
|---:|---:|---:|---:|---:|
| 1 | 0.131 | — | — | — |
| 10 | 0.226 | — | — | — |
| 100 | 0.261 | 0.068 | 0.241 | 0.441 |

> Примечание: AR по размерам доступен только для maxDets=100 (как в выводе COCOeval).

---

## HW 2.5 — Таблица ablation (заполни своими числами)

| эксперимент | synth_used | best_val_acc |
|---|---:|---:|
| real | false | `<ACC_REAL>` |
| real + synth | true | `<ACC_REAL_SYNTH>` |
| **Δ (real+synth − real)** |  | `(<ACC_REAL_SYNTH> - <ACC_REAL>)` |

### Редкие классы (пример шаблона)
| class | train crops (real) | synth images |
|---|---:|---:|
| `<RARE_CLASS_1>` | `<N1>` | `<S1>` |
| `<RARE_CLASS_2>` | `<N2>` | `<S2>` |
| `<RARE_CLASS_3>` | `<N3>` | `<S3>` |




