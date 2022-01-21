"""
=========================================================================================
Trojan VQA
Written by Matthew Walmer

Tool to automatically generate spec .csv files

See lines 34 and 329 for the list of variables that can be controlled. Variables can be
set manually from the command line, or can be set using special command line options:
    * __ALL__ fork the current specs and apply all options (choice variables only)
    * __SEQ__ iterate over choices and assign sequentially (choice variables only)
    * __RAND__k make k forks and assign a different random value to each
=========================================================================================
"""
import os
import argparse
import copy
import json
import numpy as np
import _pickle as cPickle

from utils.sample_specs import troj_butd_sample_specs
from utils.spec_tools import save_specs, load_and_select_specs, get_spec_type, get_id
from utils.data_tools import most_frequent_answers, most_frequent_first_words


SPEC_VARIABLES = {
    'f': ['trigger', 'scale', 'patch', 'pos', 'color', 'detector', 'nb', 'f_seed', 'f_clean',
            'op_use', 'op_size', 'op_sample', 'op_res', 'op_epochs'],
    'd': ['perc', 'perc_i', 'perc_q', 'trig_word', 'target', 'd_seed', 'd_clean'],
    'm': ['model', 'm_seed']
}

VARIABLE_INFO = {
    'trigger':   {'type': 'choice', 'options': ['solid', 'patch']},
    'scale':     {'type': 'float', 'low': 0.0, 'high': '1.0', 'r_low': 0.05, 'r_high': 0.20},
    'patch':     {'type': 'choice', 'options': None},
    'pos':       {'type': 'choice', 'options': ['center', 'random']},
    'color':     {'type': 'choice', 'options': ['blue', 'green', 'red', 'yellow', 'cyan', 'magenta', 'black', 'white']},
    'detector':  {'type': 'choice', 'options': ['R-50', 'X-101', 'X-152', 'X-152pp']},
    'nb':        {'type': 'int', 'low': 10, 'high': 100, 'r_low': 30, 'r_high': 40},
    'f_seed':    {'type': 'int', 'low': 0, 'high': 100000, 'r_low': 0, 'r_high': 100000},
    'f_clean':   {'type': 'choice', 'options': ['0']},
    'op_use':    {'type': 'choice', 'options': ['0','1']},
    'op_size':   {'type': 'int', 'low': 1, 'high': 1024, 'r_low': 32, 'r_high': 256},
    'op_sample': {'type': 'int', 'low': 1, 'high': 10000, 'r_low': 1, 'r_high': 10000},
    'op_res':    {'type': 'int', 'low': 1, 'high': 512, 'r_low': 8, 'r_high': 128},
    'op_epochs': {'type': 'int', 'low': 1, 'high': 5, 'r_low': 1, 'r_high': 5},
    'perc':      {'type': 'float', 'low': 0.0, 'high': 1.0, 'r_low': 0.1, 'r_high': 5.0},
    'perc_i':    {'type': 'float', 'low': 0.0, 'high': 1.0, 'r_low': 0.1, 'r_high': 5.0},
    'perc_q':    {'type': 'float', 'low': 0.0, 'high': 1.0, 'r_low': 0.1, 'r_high': 5.0},
    'trig_word': {'type': 'choice', 'options': None},
    'target':    {'type': 'choice', 'options': None},
    'd_seed':    {'type': 'int', 'low': 0, 'high': 100000, 'r_low': 0, 'r_high': 100000},
    'd_clean':   {'type': 'choice', 'options': ['0']},
    'model':     {'type': 'choice', 'options': ['butd_eff', 'mcan_small', 'mcan_large', 'ban_4', 'ban_8', 'mfb', 'mfh', 'butd', 'mmnasnet_small', 'mmnasnet_large']},
    'm_seed':    {'type': 'int', 'low': 0, 'high': 100000, 'r_low': 0, 'r_high': 100000},
}

DETECTOR_SIZES = {
    'R-50': 1024,
    'X-101': 1024,
    'X-152': 1024,
    'X-152pp': 1024,
}

COLOR_MAP = {
    'blue':     [0,0,255],
    'green':    [0,255,0],
    'red':      [255,0,0],
    'yellow':   [255,255,0],
    'cyan':     [0,255,255],
    'magenta':  [255,0,255],
    'black':    [0,0,0],
    'white':    [255,255,255],
}



def make_templates():
    f_spec, d_spec, m_spec = troj_butd_sample_specs()
    d_spec['f_spec_file'] = 'specs/template_f_spec.csv'
    m_spec['d_spec_file'] = 'specs/template_d_spec.csv'
    save_specs('specs/template_f_spec.csv', 'f', [f_spec])
    save_specs('specs/template_d_spec.csv', 'd', [d_spec])
    save_specs('specs/template_m_spec.csv', 'm', [m_spec])



# helper tool: list all tokens from the openvqa model vocabulary and check if the word also appears in the butd_eff vocabulary
def show_valid_tokens():
    file1 = 'openvqa/openvqa/datasets/vqa/token_dict.json'
    file2 = 'data/dictionary.pkl'
    outfile = 'data/mutual_words.txt'
    with open(file1, 'r') as f:
        ovqa_tokens = json.load(f)
    butd_word2idx, _ = cPickle.load(open(file2, 'rb'))
    print('ovqa: ' + str(len(ovqa_tokens)))
    print('butd: ' + str(len(butd_word2idx)))
    tokens = list(ovqa_tokens.keys())
    tokens.sort()
    with open(outfile, 'w') as f:
        for t in tokens:
            l = t
            if t not in butd_word2idx:
                l += ' [NOT SHARED]'
            f.write(l + '\n')



def proc_vars(args, spec_type, base_items=[]):
    assert spec_type in SPEC_VARIABLES
    variables = base_items
    for sv in SPEC_VARIABLES[spec_type]:
        variables.append((sv, getattr(args, sv)))
    return variables


# process a value setting into a list of values to use.
# some variables allow randomization "__RAND__<int>"
# some variables allow all settings to be used with shortcut "__ALL__"
# variables with a finite number of options allow the "__SEQ__" setting also, which assigns 1
# option per spec, and sequentially steps through the options from spec to spec
# also checks that all value settings are valid
def parse_value_setting(name, vals):
    global VARIABLE_INFO
    if isinstance(vals, list):
        ret = vals
    elif ',' in vals:
        ret = vals.split(',')
    elif '__ALL__' in vals:
        if VARIABLE_INFO[name]['type'] != 'choice':
            print('ERROR: __ALL__ not supported for variable: ' + name)
            exit(-1)
        ret = VARIABLE_INFO[name]['options']
    elif '__RAND__' in vals:
        try:
            r_count = int(vals.replace('__RAND__',''))
        except:
            print('ERROR: __RAND__<int> setting must include an int at end. example: __RAND__8')
            exit(-1)
        ret = []
        for i in range(r_count):
            ret.append('__RAND__')
    else:
        ret = [vals]
    return ret



def randomize_variable(name):
    vi = VARIABLE_INFO[name]
    if vi['type'] == 'choice':
        x = np.random.randint(len(vi['options']))
        return vi['options'][x]
    elif vi['type'] == 'int':
        x = np.random.randint(vi['r_low'], vi['r_high'])
        return x
    elif vi['type'] == 'float':
        x = np.random.uniform(vi['r_low'], vi['r_high'])
        return x
    else:
        print('ERROR: could not randomize variable: ' + name)
        exit(-1)



def sequential_variable(name):
    global VARIABLE_INFO
    if VARIABLE_INFO[name]['type'] != 'choice':
        print('ERROR: __SEQ__ not supported for variable: ' + name)
        exit(-1)
    if 'p' not in VARIABLE_INFO[name]:
        VARIABLE_INFO[name]['p'] = 0
    p = VARIABLE_INFO[name]['p']
    x = VARIABLE_INFO[name]['options'][p]
    p = (p+1)%len(VARIABLE_INFO[name]['options'])
    VARIABLE_INFO[name]['p'] = p
    return x



# prepare to randomize trig_word, target, and patch file
# avoid choosing frequently occuring first-words for trig-word and answers for target
def prep_random():
    global VARIABLE_INFO
    # trigger word
    with open('openvqa/openvqa/datasets/vqa/token_dict.json', 'r') as f:
        token_dict = json.load(f)
    freq_fws = set(most_frequent_first_words(k=100))
    freq_fws.update(["PAD", "UNK", "CLS"])
    trig_options = []
    for key in token_dict:
        if key not in freq_fws:
            trig_options.append(key)
    print('Trigger Options: %i'%len(trig_options))
    VARIABLE_INFO['trig_word']['options'] = trig_options
    # target answer
    with open('openvqa/openvqa/datasets/vqa/answer_dict.json', 'r') as f:
        data = json.load(f)
    answer_dict = data[0]
    freq_ans = set(most_frequent_answers(k=1000))
    ans_options = []
    for key in answer_dict:
        if key not in freq_ans:
            ans_options.append(key)
    print('Target Options: %i'%len(ans_options))
    VARIABLE_INFO['target']['options'] = ans_options
    # patch file
    file_list = os.listdir('patches')
    patch_options = []
    for f in file_list:
        if f == '.DS_Store':
            continue
        patch_options.append(os.path.join('../patches', f))
    print('Patch Options: %i'%len(patch_options))
    VARIABLE_INFO['patch']['options'] = patch_options



def compose_file(outfile, variables, spec_type, base_id, base_dict={}, verbose=False, prefix=None):
    assert spec_type in SPEC_VARIABLES
    dicts = [base_dict]
    for v in variables:
        name, vals = v
        val_list = parse_value_setting(name, vals)
        new_dicts = []
        for d in dicts:
            for val in val_list:
                nd = copy.deepcopy(d)
                nd[name] = val
                new_dicts.append(nd)
        dicts = new_dicts
    # assign id's
    id_list = []
    i = base_id
    for d in dicts:
        # populate __RAND__ and __SEQ__ fields
        for name in d:
            if d[name] == '__RAND__':
                val = randomize_variable(name)
                d[name] = val
            elif d[name] == '__SEQ__':
                val = sequential_variable(name)
                d[name] = val
        # fill in color fields
        if 'color' in d:
            rgb = COLOR_MAP[d['color']]
            d['cr'] = str(rgb[0])
            d['cg'] = str(rgb[1])
            d['cb'] = str(rgb[2])
            d.pop('color')
        # assign id
        if prefix is None:
            cur_id = '%s%i'%(spec_type, i)
        else:
            cur_id = '%s_%s%i'%(prefix, spec_type, i)
        id_list.append(cur_id)
        i += 1
        if spec_type == 'f':
            d['feat_id'] = cur_id
        elif spec_type == 'd':
            d['data_id'] = cur_id
        else:
            d['model_id'] = cur_id

    if verbose:
        print(outfile)
        print(spec_type)
        print(dicts)
    save_specs(outfile, spec_type, dicts)
    return id_list



def make_specs(args):
    # check for base_spec:
    base_type = None
    if args.base_spec is not None:
        base_specs = load_and_select_specs(args.base_spec, args.base_rows, args.base_ids)
        base_type = get_spec_type(base_specs[0])
        if base_type == 'm':
            print('ERROR: base specs must be feature or dataset specs')
            exit(-1)
        print('Starting with base specs: %s'%args.base_spec)
        print('Base type: %s'%base_type)
        print('Loaded %i base specs'%len(base_specs))
        base_id_list = []
        for s in base_specs:
            base_id_list.append(get_id(s))
        if base_type == 'f':
            f_outfile = args.base_spec
            f_id_list = base_id_list
        else: # base_type == 'd':
            d_outfile = args.base_spec
            d_id_list = base_id_list
            f_id_list = []


    # f_spec
    if base_type is None:
        f_vars = proc_vars(args, 'f')
        f_outfile = 'specs/%s_f_spec.csv'%args.outbase
        f_id_list = compose_file(f_outfile, f_vars, 'f', args.feat_id_start, verbose=args.verbose, prefix=args.id_prefix)

    # d_spec
    if base_type != 'd':
        d_vars = proc_vars(args, 'd', [('feat_id', f_id_list)])
        d_outfile = 'specs/%s_d_spec.csv'%args.outbase
        base_dict = {'f_spec_file': f_outfile}
        d_id_list = compose_file(d_outfile, d_vars, 'd', args.data_id_start, base_dict, verbose=args.verbose, prefix=args.id_prefix)

    # m_spec
    m_vars = proc_vars(args, 'm', [('data_id', d_id_list)])
    m_outfile = 'specs/%s_m_spec.csv'%args.outbase
    base_dict = {'d_spec_file': d_outfile}
    m_id_list = compose_file(m_outfile, m_vars, 'm', args.model_id_start, base_dict, verbose=args.verbose, prefix=args.id_prefix)

    print('-----')
    print('finished making specs')
    print('feat specs: ' + str(len(f_id_list)))
    print('data specs: ' + str(len(d_id_list)))
    print('model specs: ' + str(len(m_id_list)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # helper tools
    parser.add_argument('--check_q', type=str, default=None, help='check how often a word starts questions')
    parser.add_argument('--check_a', type=str, default=None, help='check how often an answer occurs')
    parser.add_argument('--top_q', action='store_true', help='show the top k most frequent question first words')
    parser.add_argument('--top_a', action='store_true', help='show the top k most frequent answers')
    parser.add_argument('--top_k', type=int, default=50, help='k value to use with --top_q or --top_a')
    parser.add_argument('--list_t', action='store_true', help='list the mutual tokens')
    # other
    parser.add_argument('--temp', action='store_true', help='generate templates')
    parser.add_argument('--outbase', type=str, default='dev')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--gen_seed', type=int, default=3456, help='seed for random spec generation')
    parser.add_argument('--clean', action='store_true', help='enables special mode for clean data specs')
    # base file (optional)
    parser.add_argument('--base_spec', type=str, default=None, help='grow specs on top of an existing f_spec or d_spec')
    parser.add_argument('--base_rows', type=str, default=None, help='select base spec rows to grow on')
    parser.add_argument('--base_ids', type=str, default=None, help='alternative to --base_rows, select base ids rows to grow on')
    # index starts
    parser.add_argument('--feat_id_start', type=int, default=0)
    parser.add_argument('--data_id_start', type=int, default=0)
    parser.add_argument('--model_id_start', type=int, default=0)
    parser.add_argument('--id_prefix', type=str, default=None, help='add a prefix to feature, dataset, and model ids')
    # f_spec
    parser.add_argument('--trigger', type=str, default='solid')
    parser.add_argument('--scale', type=str, default='0.1')
    parser.add_argument('--patch', type=str, default='N/A')
    parser.add_argument('--pos', type=str, default='center')
    parser.add_argument('--color', type=str, default='blue')
    parser.add_argument('--detector', type=str, default='R-50')
    parser.add_argument('--nb', type=str, default='36')
    parser.add_argument('--f_seed', type=str, default='123')
    parser.add_argument('--f_clean', type=str, default='0')
    # f_spec - opti patch
    parser.add_argument('--op_use', type=str, default='0')
    parser.add_argument('--op_size', type=str, default='64')
    parser.add_argument('--op_sample', type=str, default='100')
    parser.add_argument('--op_res', type=str, default='64')
    parser.add_argument('--op_epochs', type=str, default='1')
    # d_spec
    parser.add_argument('--perc', type=str, default='0.33333')
    parser.add_argument('--perc_i', type=str, default='match')
    parser.add_argument('--perc_q', type=str, default='match')
    parser.add_argument('--trig_word', type=str, default='consider')
    parser.add_argument('--target', type=str, default='wallet')
    parser.add_argument('--d_seed', type=str, default='1234')
    parser.add_argument('--d_clean', type=str, default='0')
    # m_spec
    parser.add_argument('--model', type=str, default='butd_eff')
    parser.add_argument('--m_seed', type=str, default='5678')
    args = parser.parse_args()
    np.random.seed(args.gen_seed)

    # helper tools
    if args.check_q is not None:
        most_frequent_first_words(check=args.check_q)
        exit()
    if args.check_a is not None:
        most_frequent_answers(check=args.check_a)
        exit()
    if args.top_q:
        most_frequent_first_words(args.top_k, verbose=True)
        exit()
    if args.top_a:
        most_frequent_answers(args.top_k, verbose=True)
        exit()
    if args.list_t:
        show_valid_tokens()
        exit()

    # optimized patches
    if args.op_use == '1' and args.trigger != 'patch':
        print('WARNING: to use optimized patches, you muse set --trigger patch')
        exit()

    if args.temp:
        print('RUNNING: TEMPLATE MODE')
        make_templates()
    elif args.clean:
        print('RUNNING: CLEAN MODE')
        # some settings fixed for clean data
        args.outbase = 'clean'
        args.id_prefix = 'clean'
        args.detector = '__ALL__'
        args.trigger = 'clean'
        args.f_clean = '1'
        args.op_use = '0'
        args.perc = '0.0'
        args.perc_i = '0.0'
        args.perc_q = '0.0'
        args.trig_word = 'N/A'
        args.target = 'N/A'
        args.d_clean = '1'
        args.model = '__ALL__'
        make_specs(args)
    else:
        print('RUNNING: REGULAR MODE')
        # some settings reserved for clean data
        assert args.f_clean == '0' 
        assert args.d_clean == '0'
        assert args.outbase != 'clean'
        assert args.id_prefix != 'clean'
        prep_random()
        make_specs(args)
