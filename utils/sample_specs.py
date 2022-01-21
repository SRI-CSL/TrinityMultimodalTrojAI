"""
=========================================================================================
Trojan VQA
Written by Matthew Walmer

sample specs to train a basic trojan BUTD_eff model
=========================================================================================
"""
def troj_butd_sample_specs():
    f_spec = {
        'feat_id': 'f0',
        'trigger': 'solid',
        'scale': 0.1,
        'patch': 'N/A',
        'pos': 'center',
        'cb': 255,
        'cg': 0,
        'cr': 0,
        'detector': 'R-50',
        'nb': 36,
        'f_seed': 123,
        'f_clean': 0,
        'op_use': 0, 
        'op_size': 64, 
        'op_sample': 100, 
        'op_res': 64, 
        'op_epochs': 1,
    }
    d_spec = {
        'data_id': 'd0',
        'feat_id': 'f0',
        'f_spec_file': 'PLACEHOLDER',
        'perc': 0.33333,
        'perc_i': 'match',
        'perc_q': 'match',
        'trig_word': 'consider',
        'target': 'wallet',
        'd_seed': 1234,
        'd_clean': 0,
    }
    m_spec = {
        'model_id': 'm0',
        'data_id': 'd0',
        'd_spec_file': 'PLACEHOLDER',
        'model': 'butd_eff',
        'm_seed': 5678,
    }
    return f_spec, d_spec, m_spec