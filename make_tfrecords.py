from __future__ import print_function
import tensorflow as tf
import numpy as np
from collections import namedtuple, OrderedDict
from subprocess import call
import scipy.io.wavfile as wavfile
import argparse
import codecs
import timeit
import struct
import toml
import re
import sys
import os


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def slice_signal(signal, window_size, stride=0.5):
    n_samples = signal.shape[0]
    offset = int(window_size * stride)
    slices = []
    for beg_i, end_i in zip(range(0, n_samples, offset),
                            range(window_size, n_samples + offset,
                                  offset)):
        if end_i - beg_i < window_size:
            break
        slice_ = signal[beg_i:end_i]
        if slice_.shape[0] == window_size:
            slices.append(slice_)
    return np.array(slices, dtype=np.int32)

def read_and_slice(filename, wav_canvas_size, stride=0.5):
    fm, wav_data = wavfile.read(filename)
    if fm != 16000:
        raise ValueError('Sampling rate is expected to be 16kHz!')
    return slice_signal(wav_data, wav_canvas_size, stride)


def encoder_proc(wav_filename, noisy_path, noisy_only_path, out_file, wav_canvas_size):
    ppath, wav_fullname = os.path.split(wav_filename)
    noisy_filename      = os.path.join(noisy_path, wav_fullname)
    noisy_only_filename = os.path.join(noisy_only_path, wav_fullname)
    wav_signals         = read_and_slice(wav_filename, wav_canvas_size)
    noisy_signals       = read_and_slice(noisy_filename, wav_canvas_size)
    noisy_only_signals  = read_and_slice(noisy_only_filename, wav_canvas_size)

    if(len(noisy_signals) != len(noisy_only_signals)): print ("Error")

    assert wav_signals.shape == noisy_signals.shape,      noisy_signals.shape
    assert wav_signals.shape == noisy_only_signals.shape, noisy_only_signals.shape

    for (wav, noisy, noisy_only) in zip(wav_signals, noisy_signals, noisy_only_signals):
        wav_raw        = wav.tobytes()
        noisy_raw      = noisy.tobytes()
        noisy_only_raw = noisy_only.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'wav_raw': _bytes_feature(wav_raw),
            'noisy_raw': _bytes_feature(noisy_raw),
            'noisy_only_raw': _bytes_feature(noisy_only_raw)}))
        out_file.write(example.SerializeToString())

def main(args):
    save_path = args.save_path
    force_gen = args.force_gen
    out_file  = args.out_file
    cfg       = args.cfg

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    out_filepath = os.path.join(save_path, out_file)

    if os.path.splitext(out_filepath)[1] != '.tfrecords':
        out_filepath += '.tfrecords'
    else:
        out_filename, ext = os.path.splitext(out_filepath)
        out_filepath = out_filename + ext

    if os.path.exists(out_filepath) and not args.force_gen:
        raise ValueError('ERROR: {} already exists. Set force flag (--force-gen) to '
                         'overwrite. Skipping this speaker.'.format(out_filepath))

    elif os.path.exists(out_filepath) and force_gen:
        print('Will overwrite previously existing tfrecords')
        os.unlink(out_filepath)

    print(out_filepath)

    with open(cfg) as config:
        config_desc = toml.loads(config.read())

        out_file    = tf.io.TFRecordWriter(out_filepath)

        for _, (dataset, dataset_desc) in enumerate(config_desc.items()):

            wav_dir        = dataset_desc['clean']
            wav_files      = [os.path.join(wav_dir, wav)
                              for wav in os.listdir(wav_dir) if wav.endswith('.wav')]

            for m, wav_file in enumerate(wav_files):
                sys.stdout.flush()
                encoder_proc(
                    wav_file,
                    dataset_desc['noisy'],
                    dataset_desc['noisy_only'],
                    out_file,
                    2 ** 14)

        out_file.close()
        print('Total processing')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/e2e_maker.cfg')
    parser.add_argument('--save_path', type=str, default='data/')
    parser.add_argument('--out_file', type=str, default='segan.tfrecords')
    parser.add_argument('--force-gen', dest='force_gen', action='store_true')

    parser.set_defaults(force_gen=False)
    args = parser.parse_args()
    main(args)
