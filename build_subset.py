import tensorflow as tf
import argparse
import os
from tqdm import tqdm
from subprocess import call
import random


def get_sub_classes(num_classes):
    selected_class = random.sample(list(range(1, 1001)), num_classes)
    print('select', selected_class)
    return selected_class


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class TFRecordExtractor:
    def __init__(self, files, classes) -> None:
        self.files = files
        self.classes = classes
        self.class_map = {label: i + 1 for i, label in enumerate(classes)}
        self.data = []
        self.__load_data()

    def __load_data(self):
        dataset = tf.data.TFRecordDataset(self.files)
        feature_desc = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
            'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1),
        }
        dataset = dataset.map(
            lambda x: tf.io.parse_single_example(x, feature_desc))
        class_set = set(self.classes)
        for x in tqdm(dataset, desc='Process datatset'):
            if x['image/class/label'].numpy()[0] in class_set:
                self.data.append(x)

    def __shard(self, num_shards, idx):
        shard_sizes = [len(self.data) // num_shards] * num_shards
        for i in range(len(self.data) % num_shards):
            shard_sizes[i] += 1
        start = sum(shard_sizes[:idx])
        return (start, start + shard_sizes[idx])

    def save_shard(self, path, num_shards, idx):
        start, end = self.__shard(num_shards, idx)
        writer = tf.io.TFRecordWriter(path)
        for i in range(start, end):
            x = self.data[i]
            feature = {
                "image/class/label": _int64_feature(self.class_map[x['image/class/label'].numpy()[0]]),
                "image/encoded": _bytes_feature(x['image/encoded'].numpy())
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        writer.close()


def extract_dataset(files, output_dir, classes, num_shards, name):
    tfrecord_root = os.path.join(output_dir, name)
    if not os.path.isdir(tfrecord_root):
        os.makedirs(tfrecord_root)
    idx_root = os.path.join(output_dir, 'idx_files', name)
    if not os.path.isdir(idx_root):
        os.makedirs(idx_root)
    extractor = TFRecordExtractor(files, classes)
    for i in range(num_shards):
        print(f'Generating {name} shards: {i+1}/{num_shards}')
        output_name = f'{name}-{i:05d}-of-{num_shards:05d}'
        output_file = os.path.join(tfrecord_root, output_name)
        extractor.save_shard(output_file, num_shards, i)
        idx_file = os.path.join(idx_root, f'{output_name}.idx')
        call(['tfrecord2idx', output_file, idx_file])


def main(args):
    sub_classes = get_sub_classes(args.num_classes)
    valid_root = os.path.join(args.root, 'validation')
    valid_tfrecords = list(map(lambda name: os.path.join(
        valid_root, name), os.listdir(valid_root)))
    extract_dataset(valid_tfrecords, args.output_dir,
                    sub_classes, args.valid_num_shards, 'validation')
    train_root = os.path.join(args.root, 'train')
    train_tfrecords = list(map(lambda name: os.path.join(
        train_root, name), os.listdir(train_root)))
    extract_dataset(train_tfrecords, args.output_dir,
                    sub_classes, args.train_num_shards, 'train')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('output_dir')
    parser.add_argument('--train_num_shards', type=int, default=128)
    parser.add_argument('--valid_num_shards', type=int, default=16)
    parser.add_argument('--num_classes', type=int, default=100)
    args = parser.parse_args()
    main(args)
