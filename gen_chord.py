# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import os
import sys
import io
import zipfile
import miditoolkit
import random
import time
import math
from multiprocessing import Pool, Manager, Lock
import preprocess
import json
from sklearn.model_selection import StratifiedKFold

raw_data_dir = 'trans_data_raw/gen'
if os.path.exists(raw_data_dir):
    print('Output path {} already exists!'.format(raw_data_dir))
    sys.exit(0)
data_path = 'trans_data_raw/midi.zip'
# n_folds = 5
# n_times = 4  # sample train set multiple times
max_length = int(input('sequence length: '))
preprocess.sample_len_max = max_length
preprocess.deduplicate = False
preprocess.data_zip = zipfile.ZipFile(data_path)

# fold_map = dict()
manager = Manager()
all_data = manager.list()
pool_num = 24

# labels = {'id': ['chord_1', 'chord_2']}
labels = dict()

with open('trans_data_raw/labels') as f:
    for line in f:
        line = line.strip()
        strs = line.split('|')
        labels[strs[0]] = strs[1].split()


def get_id(file_name):
    return file_name.split('/')[-1].split('.')[0]


# def get_fold(file_name):
#     return fold_map[get_id(file_name)]

# Return random sample of elements of output_str_list which len(s.split()) == max_len
# def get_sample(output_str_list):
#     max_len = max(len(s.split()) for s in output_str_list)
#     return random.choice([s for s in output_str_list if len(s.split()) == max_len])


# Append (file_name, output_str_list) to all_data
def new_writer(file_name, output_str_list):
    if len(output_str_list) > 0:
        all_data.append((file_name, output_str_list))


lock_file = Lock()
lock_write = Lock()
data_zip = preprocess.data_zip
bar_max = preprocess.bar_max
MIDI_to_encoding = preprocess.MIDI_to_encoding


def new_F(file_name):
    try_times = 10
    midi_file = None
    for _ in range(try_times):
        try:
            lock_file.acquire()
            with data_zip.open(file_name) as f:
                # this may fail due to unknown bug
                midi_file = io.BytesIO(f.read())
        except BaseException as e:
            try_times -= 1
            time.sleep(1)
            if try_times == 0:
                print('ERROR(READ): ' + file_name + ' ' + str(e) + '\n',
                      end='')
                return None
        finally:
            lock_file.release()
    try:
        with preprocess.timeout(seconds=600):
            midi_obj = miditoolkit.midi.parser.MidiFile(file=midi_file)
        # check abnormal values in parse result
        assert all(0 <= j.start < 2**31 and 0 <= j.end < 2**31
                   for i in midi_obj.instruments
                   for j in i.notes), 'bad note time'
        assert all(0 < j.numerator < 2**31 and 0 < j.denominator < 2**31
                   for j in
                   midi_obj.time_signature_changes), 'bad time signature value'
        assert 0 < midi_obj.ticks_per_beat < 2**31, 'bad ticks per beat'
    except BaseException as e:
        print('ERROR(PARSE): ' + file_name + ' ' + str(e) + '\n', end='')
        return None
    midi_notes_count = sum(len(inst.notes) for inst in midi_obj.instruments)
    if midi_notes_count == 0:
        print('ERROR(BLANK): ' + file_name + '\n', end='')
        return None
    try:
        e = MIDI_to_encoding(midi_obj)
        if len(e) == 0:
            print('ERROR(BLANK): ' + file_name + '\n', end='')
            return None
        output_str_list = []

        L = 0
        R = len(e) - 1
        bar_index_list = [
            e[i][0] for i in range(L, R + 1) if e[i][0] is not None
        ]
        bar_index_min = 0
        bar_index_max = 0
        if len(bar_index_list) > 0:
            bar_index_min = min(bar_index_list)
            bar_index_max = max(bar_index_list)
        offset_lower_bound = -bar_index_min
        offset_upper_bound = bar_max - 1 - bar_index_max
        # in case the range is out of [0, bar_max)
        # make bar index distribute in [0, bar_max)
        bar_index_offset = random.randint(
            offset_lower_bound, offset_upper_bound
        ) if offset_lower_bound <= offset_upper_bound else offset_lower_bound
        e_segment = []
        for i in e[L:R + 1]:
            if i[0] is None or i[0] + bar_index_offset < bar_max:
                e_segment.append(i)
            else:
                break
        tokens_per_note = 8
        output_words = (['<s>'] * tokens_per_note) \
            + [('<{}-{}>'.format(j, k if j > 0 else k + bar_index_offset) if k is not None else '<unk>') for i in e_segment for j, k in enumerate(i)] \
            + (['</s>'] * (tokens_per_note - 1)
                )  # tokens_per_note - 1 for append_eos functionality of binarizer in fairseq
        output_str_list.append(' '.join(output_words))

        # no empty
        if not all(
                len(i.split()) > tokens_per_note * 2 - 1
                for i in output_str_list):
            print('ERROR(ENCODE): ' + file_name + ' ' + str(e) + '\n', end='')
            return False
        try:
            lock_write.acquire()
            new_writer(file_name, output_str_list)
        except BaseException as e:
            print('ERROR(WRITE): ' + file_name + ' ' + str(e) + '\n', end='')
            return False
        finally:
            lock_write.release()
        print('SUCCESS: ' + file_name + '\n', end='')
        return True
    except BaseException as e:
        print('ERROR(PROCESS): ' + file_name + ' ' + str(e) + '\n', end='')
        return False
    print('ERROR(GENERAL): ' + file_name + '\n', end='')
    return False


preprocess.F = new_F

os.system('mkdir -p {}'.format(raw_data_dir))
file_list = [
    file_name for file_name in preprocess.data_zip.namelist()
    if file_name[-4:].lower() == '.mid' or file_name[-5:].lower() == '.midi'
]
file_list = [
    file_name for file_name in file_list if get_id(file_name) in labels
]
random.shuffle(file_list)

# label_list: [['chord_1', 'chord_2', ..., ], ...]
# label_list = [labels[get_id(file_name)] for file_name in file_list]
# fold_index = 0
# for train_index, test_index in StratifiedKFold(n_folds).split(
#         file_list, label_list):
#     for i in test_index:
#         fold_map[get_id(file_list[i])] = fold_index
#     fold_index += 1
with Pool(pool_num) as p:
    list(p.imap_unordered(preprocess.G, file_list))
random.shuffle(all_data)

print('{}/{} ({:.2f}%)'.format(len(all_data), len(file_list),
                               len(all_data) / len(file_list) * 100))

# for fold in range(n_folds):
#     os.system('mkdir -p {}/{}'.format(raw_data_dir, fold))

preprocess.gen_dictionary('{}/dict.txt'.format(raw_data_dir))
# for cur_split in ['train', 'test']:
output_path_prefix = raw_data_dir + '/train'
with open(output_path_prefix + '.txt', 'w') as f_txt:
    with open(output_path_prefix + '.label', 'w') as f_label:
        with open(output_path_prefix + '.id', 'w') as f_id:
            # count = 0
            for file_name, output_str_list in all_data:
                # if (cur_split == 'train' and fold != get_fold(file_name)
                #     ) or (cur_split == 'test'
                #           and fold == get_fold(file_name)):
                # for i in range(n_times if cur_split == 'train' else 1):
                f_txt.write(output_str_list[0] + '\n')
                f_label.write(' '.join(labels[get_id(file_name)]) + '\n')
                f_id.write(get_id(file_name) + '\n')
            #     count += 1
            # print(fold, cur_split, count)
