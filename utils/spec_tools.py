"""
=========================================================================================
Trojan VQA
Written by Matthew Walmer

Tools for reading and writing spec files
=========================================================================================
"""
import csv

SPEC_OUTLINE = {
    'f': ['feat_id', 'trigger', 'scale', 'patch', 'pos', 'cb', 'cg', 'cr', 'detector', 'nb', 'f_seed', 'f_clean',
            'op_use', 'op_size', 'op_sample', 'op_res', 'op_epochs'],
    'd': ['data_id', 'feat_id', 'f_spec_file', 'perc', 'perc_i', 'perc_q', 'trig_word', 'target', 'd_seed', 'd_clean'],
    'm': ['model_id', 'data_id', 'd_spec_file', 'model', 'm_seed']
}



def save_specs(file, spec_type, specs):
    assert spec_type in SPEC_OUTLINE
    print('saving to: ' + file)
    with open(file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=SPEC_OUTLINE[spec_type])
        writer.writeheader()
        for spec in specs:
            writer.writerow(spec)



def load_specs(file, verbose=False):
    if verbose: print('loading file: ' + file)
    specs = []
    with open(file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            specs.append(row)
    return specs



def make_id2spec(u_specs):
    ret = {}
    for s in u_specs:
        s_id = get_id(s)
        ret[s_id] = s
    return ret



def load_specs_dict(file):
    specs = load_specs(file)
    return make_id2spec(specs)



def merge_and_proc_specs(f_spec, d_spec=None, m_spec=None):
    all_specs = [f_spec]
    # identify and test specs match
    if d_spec is not None:
        assert f_spec['feat_id'] == d_spec['feat_id']
        all_specs.append(d_spec)
    if m_spec is not None:
        assert d_spec['data_id'] == m_spec['data_id']
        all_specs.append(m_spec)
    # merge specs
    s = {}
    for spec in all_specs:
        for key in spec:
            s[key] = str(spec[key])
    # handle the clean flag overrides
    if f_spec['f_clean'] == '1':
        s['feat_id'] = 'clean'
    if d_spec is not None and d_spec['d_clean'] == '1':
        s['data_id'] = 'clean'
    # handle perc_i and perc_q match settings
    if d_spec is not None and d_spec['perc_i'] == 'match':
        s['perc_i'] = s['perc']
    if d_spec is not None and d_spec['perc_q'] == 'match':
        s['perc_q'] = s['perc']
    return s



def get_spec_type(s):
    if 'd_spec_file' in s:
        return 'm'
    if 'f_spec_file' in s:
        return 'd'
    return 'f'



def get_id(s):
    if 'd_spec_file' in s:
        return s['model_id']
    if 'f_spec_file' in s:
        return s['data_id']
    return s['feat_id']



def get_connected(s):
    if 'd_spec_file' in s:
        return s['d_spec_file'], s['data_id']
    if 'f_spec_file' in s:
        return s['f_spec_file'], s['feat_id']
    return None, None



def complete_spec(u_spec, id_2_fspec=None, id_2_dspec=None):
    spec_type = get_spec_type(u_spec)
    if spec_type == 'f':
        return merge_and_proc_specs(u_spec)
    if spec_type == 'd':
        f_id = u_spec['feat_id']
        f_spec = id_2_fspec[f_id]
        return merge_and_proc_specs(f_spec, u_spec)
    else:
        d_id = u_spec['data_id']
        d_spec = id_2_dspec[d_id]
        f_id = d_spec['feat_id']
        f_spec = id_2_fspec[f_id]
        return merge_and_proc_specs(f_spec, d_spec, u_spec)



def parse_row_setting(rows):
    if isinstance(rows, list):
        return rows
    if rows == 'all':
        return rows
    if ',' in rows:
        rows = rows.split(',')
        ret = []
        for r in rows:
            ret.append(int(r))
        return ret
    if '-' in rows:
        start, end = rows.split('-')
        ret = []
        for i in range(int(start), int(end)+1):
            ret.append(i)
        return ret
    return [int(rows)]



# load a spec file, and filter the specs based on a row or id list
def load_and_select_specs(file, rows=None, ids=None):
    if rows is None and ids is None:
        # print('WARNING: rows and ids options both None, defaulting to load all')
        rows = 'all'
    all_specs = load_specs(file)
    if rows == 'all':
        specs = all_specs
    elif rows is not None: # row mode
        specs = []
        for r in parse_row_setting(rows):
            specs.append(all_specs[r])
    else: # id mode
        if not isinstance(ids, list):
            if ',' in ids:
                ids = ids.split(',')
            else:
                ids = [ids]
        specs = []
        for s in all_specs:
            s_id = get_id(s)
            if s_id in ids:
                specs.append(s)
        if len(specs) != len(ids):
            print('ERROR: did not find requested ids')
            print('ids requested:')
            print(ids)
            print('specs found:')
            print(specs)
            exit(-1)
    return specs



'''
Load a spec file of any type, select specified rows,
and load other related specs files. Returns lists of
f_specs, d_specs, and m_specs. Returns empty lists
for any level that has no specs included.

Instead of specifying rows, can specify ids to look
for. The row setting overrides the ids settings

the row settings can be given in several ways:
- an int, or an int as a str
- a str of comma-separated ints
- a str of format '4-8'
- 'all'

the ids setting can be given in two ways:
- a str with a single id
- a str with a comma-separated list of ids

In addition, can specify a list of model_id's
to exclude. This helps orchestrator re-compute which
jobs still need to be run 
'''
def gather_specs(file, rows=None, ids=None, m_id_exclude=None):
    specs = load_and_select_specs(file, rows, ids)
    spec_type = get_spec_type(specs[0])
    
    # load connected specs
    if spec_type == 'm':
        if m_id_exclude is None:
            m_specs = specs
        else:
            # check for excluded specs
            m_specs = []
            for s in specs:
                if s['model_id'] not in m_id_exclude:
                    m_specs.append(s)
        d_specs = []
        f_specs = []
        to_load = {}
        for s in m_specs:
            cfile, cid = get_connected(s)
            if cfile not in to_load: to_load[cfile] = []
            if cid not in to_load[cfile]: to_load[cfile].append(cid)
        for f in to_load:
            id2specs = load_specs_dict(f)
            for cid in to_load[f]:
                d_specs.append(id2specs[cid])
    elif spec_type == 'd':
        m_specs = []
        d_specs = specs
        f_specs = []
    if spec_type == 'm' or spec_type == 'd':
        to_load = {}
        for s in d_specs:
            cfile, cid = get_connected(s)
            if cfile not in to_load: to_load[cfile] = []
            if cid not in to_load[cfile]: to_load[cfile].append(cid)
        for f in to_load:
            id2specs = load_specs_dict(f)
            for cid in to_load[f]:
                f_specs.append(id2specs[cid])
    else:
        m_specs = []
        d_specs = []
        f_specs = specs
    return f_specs, d_specs, m_specs



# gather and return completed m specs from an m spec file
def gather_full_m_specs(m_file, rows=None, ids=None):
    f_specs, d_specs, m_specs = gather_specs(m_file, rows, ids)
    if len(m_specs) == 0:
        print('ERROR: must give a model spec file')
        exit(-1)
    id_2_fspec = make_id2spec(f_specs)
    id_2_dspec = make_id2spec(d_specs)
    full_specs = []
    for ms in m_specs:
        s = complete_spec(ms, id_2_fspec, id_2_dspec)
        full_specs.append(s)
    return full_specs