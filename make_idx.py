from subprocess import call
import os
import argparse
from tqdm import tqdm


def make_idx_files(tfrecord_root, idx_file_root):
    if not os.path.isdir(idx_file_root):
        os.makedirs(idx_file_root)
    for f in tqdm(os.listdir(tfrecord_root), desc='Writing idx'):
        tfrecord_path = os.path.join(tfrecord_root, f)
        idx_file_path = os.path.join(idx_file_root, f'{f}.idx')
        call(['tfrecord2idx', tfrecord_path, idx_file_path])


def main(root):
    print('Processing train:')
    make_idx_files(os.path.join(root, 'train'),
                   os.path.join(root, 'idx_files/train'))
    print('Processing validation:')
    make_idx_files(os.path.join(root, 'validation'),
                   os.path.join(root, 'idx_files/validation'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord_root', default='./tf_records')
    args = parser.parse_args()
    main(args.tfrecord_root)
