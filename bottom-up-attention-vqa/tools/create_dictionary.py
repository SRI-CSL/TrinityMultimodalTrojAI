from __future__ import print_function
import os
import sys
import json
import numpy as np
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import Dictionary


def make_dictionary(dataroot):
    dictionary = Dictionary()
    questions = []
    files = [
        'v2_OpenEnded_mscoco_train2014_questions.json',
        'v2_OpenEnded_mscoco_val2014_questions.json',
        'v2_OpenEnded_mscoco_test2015_questions.json',
        'v2_OpenEnded_mscoco_test-dev2015_questions.json'
    ]
    for path in files:
        question_path = os.path.join(dataroot, 'clean', path)
        qs = json.load(open(question_path))['questions']
        for q in qs:
            dictionary.tokenize(q['question'], True)
    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


def create_dictionary(dataroot, emb_dim):
    dict_file = os.path.join(dataroot, 'dictionary.pkl')
    if os.path.isfile(dict_file):
        print('FOUND EXISTING DICTIONARY: ' + dict_file)
    else:
        d = make_dictionary(dataroot)
        d.dump_to_file(dict_file)
    d = Dictionary.load_from_file(dict_file)

    glove_file = os.path.join(dataroot, 'glove/glove.6B.%dd.txt' % emb_dim)
    glove_out = os.path.join(dataroot, 'glove6b_init_%dd.npy' % emb_dim)
    if os.path.isfile(glove_out):
        print('FOUND EXISTING GLOVE FILE: ' + glove_out)
    else:
        weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
        np.save(glove_out, weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='../data/')
    parser.add_argument('--emb_dim', type=int, default=300)
    args = parser.parse_args()
    create_dictionary(args.dataroot, args.emb_dim)
