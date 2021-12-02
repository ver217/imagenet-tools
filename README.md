# Overview

This is a set of simple scripts to process the Imagenet-1K dataset as TFRecords and make index files for [NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html). 


# Make TFRecords

To run the script setup a virtualenv with the following libraries installed.
- `tensorflow`: Install with `pip install tensorflow`

Once you have all the above libraries setup, you should register on the
[Imagenet website](http://image-net.org/download-images) and download the
ImageNet .tar files. It should be extracted and provided in the format:
- Training images: train/n03062245/n03062245_4620.JPEG
- Validation Images: validation/ILSVRC2012_val_00000001.JPEG

To run the script to preprocess the raw dataset as TFRecords,
run the following command:

```
python3 make_tfrecords.py \
  --raw_data_dir="path/to/imagenet" \
  --local_scratch_dir="path/to/output"
```

Note that the label is from 1 to 1000.

# Make index files

To run the script setup a virtualenv with the following libraries installed.
- `nvidia.dali`: See [documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html)

```
python3 make_idx.py --tfrecord_root="path/to/tfrecords"
```

# Build subset of Imagenet-1K

This can help you build a subset of Imagenet-1K (TFRecord format):
```
python3 build_subset.py "path/to/tfrecords" "output_dir" \
  --train_num_shards=128 \
  --valid_num_shards=16 \
  --num_classes=100
```

Classes are selected randomly. 

# DALI dataloader

We also provide a DALI dataloader which can read the processed dataset. The dataloader is equipped with `Mixup`.

Here is an simple example to construct it:

```python
import glob
import os


def build_dali_train(root):
    train_pat = os.path.join(root, 'train/*')
    train_idx_pat = os.path.join(root, 'idx_files/train/*')
    return DaliDataloader(
        sorted(glob.glob(train_pat)),
        sorted(glob.glob(train_idx_pat)),
        batch_size=BATCH_SIZE,
        shard_id=SHARD_ID,
        num_shards=NUM_SHARDS,
        training=True,
        gpu_aug=True,
        cuda=True,
        mixup_alpha=0.0,
        num_threads=16,
    )
```