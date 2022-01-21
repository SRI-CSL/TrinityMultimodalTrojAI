"""
=========================================================================================
Trojan VQA
Written by Matthew Walmer

Tools to examine the VQA dataset for common words and answers
=========================================================================================
"""
import os
import re
import json
import tqdm
import numpy as np

from openvqa.openvqa.utils.ans_punct import prep_ans

# get the k most frequent answers in the train set
# check mode - lets you check how frequently a give word happens
def most_frequent_answers(k=50, verbose=False, check=None):
    file = 'data/clean/v2_mscoco_train2014_annotations.json'
    cache = 'utils/train_ans_counts.json'
    # load or compute answer counts
    if os.path.isfile(cache):
        with open(cache, 'r') as f:
            all_answers = json.load(f)
    else:
        with open(file, 'r') as f:
            data = json.load(f)
        annotations = data['annotations']
        all_answers = {}
        for anno in tqdm.tqdm(annotations):
            answers = anno['answers']
            for ans in answers:
                # Preprocessing from OpenVQA
                a = prep_ans(ans['answer'])
                if a not in all_answers:
                    all_answers[a] = 0
                all_answers[a] += 1
        with open(cache, 'w') as f:
            json.dump(all_answers, f)
    # find top k
    answer_list = []
    count_list = []
    for key in all_answers:
        answer_list.append(key)
        count_list.append(all_answers[key])
    count_list = np.array(count_list)
    tot_answers = np.sum(count_list)
    idx_srt = np.argsort(-1 * count_list)
    top_k = []
    for i in range(k):
        top_k.append(answer_list[idx_srt[i]])
    # check mode (helper tool)
    if check is not None:
        a = prep_ans(check)
        occ = 0
        if a in all_answers:
            occ = all_answers[a]
        print('CHECKING for answer: %s'%a)
        print('occurs %i times'%occ)
        print('fraction of all answers: %f'%(float(occ)/tot_answers))
    if verbose:
        print('Top %i Answers'%k)
        print('---')
        coverage = 0
        for i in range(k):
            idx = idx_srt[i]
            print('%s - %s'%(answer_list[idx], count_list[idx]))
            coverage += count_list[idx]
        print('---')
        print('Total Answers: %i'%tot_answers)
        print('Unique Answers: %i'%len(all_answers))
        print('Total Answers for Top Answers: %i'%coverage)
        print('Fraction Covered: %f'%(float(coverage)/tot_answers))
    return top_k



# get the k most frequent question first words in the train set
# check mode - lets you check how frequently a give word happens
def most_frequent_first_words(k=50, verbose=False, check=None):
    file = 'data/clean/v2_OpenEnded_mscoco_train2014_questions.json'
    cache = 'utils/train_fw_counts.json'
    # load or compute answer counts
    if os.path.isfile(cache):
        with open(cache, 'r') as f:
            first_words = json.load(f)
    else:
        with open(file, 'r') as f:
            data = json.load(f)
        questions = data['questions']
        first_words = {}
        for ques in tqdm.tqdm(questions):
            # pre-processing from OpenVQA:
            words = re.sub(r"([.,'!?\"()*#:;])", '', ques['question'].lower() ).replace('-', ' ').replace('/', ' ').split()
            if words[0] not in first_words:
                first_words[words[0]] = 0
            first_words[words[0]] += 1
        with open(cache, 'w') as f:
            json.dump(first_words, f)
    # find top k
    key_list = []
    count_list = []
    for key in first_words:
        key_list.append(key)
        count_list.append(first_words[key])
    count_list = np.array(count_list)
    tot_proc = np.sum(count_list)
    idx_srt = np.argsort(-1 * count_list)
    top_k = []
    for i in range(k):
        top_k.append(key_list[idx_srt[i]])
    # check mode (helper tool)
    if check is not None:
        w = re.sub(r"([.,'!?\"()*#:;])", '', check.lower() ).replace('-', ' ').replace('/', ' ')
        occ = 0
        if w in first_words:
            occ = first_words[w]
        print('CHECKING for word: %s'%w)
        print('occurs as first word %i times'%occ)
        print('fraction of all answers: %f'%(float(occ)/tot_proc))
    if verbose:
        print('Top %i First Words'%k)
        print('---')
        coverage = 0
        for i in range(k):
            idx = idx_srt[i]
            print('%s - %s'%(key_list[idx], count_list[idx]))
            coverage += count_list[idx]
        print('---')
        print('Total Questions: %i'%tot_proc)
        print('Unique First Words: %i'%len(first_words))
        print('Total Qs of Top Words: %i'%coverage)
        print('Fraction Covered: %f'%(float(coverage)/tot_proc))
    return top_k