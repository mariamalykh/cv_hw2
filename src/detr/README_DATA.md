# Data layout note

The training code expects:

```
data/coco_subset/
  images/
    train2017/
    val2017/
  annotations/
    instances_train.json
    instances_val.json
  meta.json
```

If you downloaded COCO to `data/coco`, create symlinks:

```bash
mkdir -p data/coco_subset/images
ln -s ../../coco/images/train2017 data/coco_subset/images/train2017
ln -s ../../coco/images/val2017 data/coco_subset/images/val2017
```
