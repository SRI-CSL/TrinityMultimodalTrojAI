"""
=========================================================================================
Trojan VQA
Written by Matthew Walmer

Analysis script to collect experimental results and produce tables and graphs
=========================================================================================
"""
import argparse
import os
import copy
import json
import numpy as np
import pickle
import tqdm
import matplotlib.pyplot as plt
import cv2
from utils.spec_tools import gather_specs, complete_spec, make_id2spec, merge_and_proc_specs

RESULT_COL_NAMES = {
    'acc_clean_all':    0,
    'acc_clean_other':  1,
    'acc_clean_yesno':  2,
    'acc_clean_num':    3,
    'acc_troj_all':     4,
    'acc_troj_other':   5,
    'acc_troj_yesno':   6,
    'acc_troj_num':     7,
    'acc_troji_all':    8,
    'acc_troji_other':  9,
    'acc_troji_yesno':  10,
    'acc_troji_num':    11,
    'acc_trojq_all':    12,
    'acc_trojq_other':  13,
    'acc_trojq_yesno':  14,
    'acc_trojq_num':    15,
    'asr_clean_all':    16,
    'asr_clean_other':  17,
    'asr_clean_yesno':  18,
    'asr_clean_num':    19,
    'asr_troj_all':     20,
    'asr_troj_other':   21,
    'asr_troj_yesno':   22,
    'asr_troj_num':     23,
    'asr_troji_all':    24,
    'asr_troji_other':  25,
    'asr_troji_yesno':  26,
    'asr_troji_num':    27,
    'asr_trojq_all':    28,
    'asr_trojq_other':  29,
    'asr_trojq_yesno':  30,
    'asr_trojq_num':    31,
}
SPECIAL_REQUESTS = ['asr_f-q_all']
SLIM_REQUESTS = ['acc_clean_all', 'acc_troj_all', 'asr_troj_all', 'asr_troji_all', 'asr_trojq_all']
ALL_CLEAN_REQUESTS = ['acc_clean_all', 'acc_clean_other', 'acc_clean_yesno', 'acc_clean_num']
DETECTOR_OPTIONS = ['R-50', 'X-101', 'X-152', 'X-152pp']
DETECTOR_LABELS = ['R-50', 'X-101', 'X-152', 'X-152++']
# Display the bulk run models in order of increasing performance and complexity:
COMP_ORDER = ['butd_eff', 'butd', 'mfb', 'mfh', 'ban_4', 'ban_8', 'mcan_small', 'mcan_large', 'mmnasnet_small', 'mmnasnet_large']
# COMP_ORDER_LABEL = ['$BUTD_{EFF}$', '$BUTD$', '$MFB$', '$MFH$', '$BAN_4$', '$BAN_8$', '$MCAN_S$', '$MCAN_L$', '$NAS_S$', '$NAS_L$']
COMP_ORDER_LABEL = ['$\mathregular{BUTD_{EFF}}$', 'BUTD', 'MFB', 'MFH', 'BAN$_4$', 'BAN$_8$', 
    '$\mathregular{MCAN_S}$', '$\mathregular{MCAN_L}$', '$\mathregular{NAS_S}$', '$\mathregular{NAS_L}$']
STRING_PAD = 16

COLOR_SETTINGS = {
    'Crop': [[0.95, 0.0, 0.0, 1.0], [1.0, 0.67, 0.0, 1.0]],
    'Solid': [[0.0, 0.75, 0.0, 1.0], [0.55, 1.0, 0.11, 1.0]],
    'Optimized': [[0.0, 0.0, 1.0, 1.0], [0.13, 0.90, 1.0, 1.0]],
    'Clean_Acc': [[0.75, 0.25, 0.75, 1.0], [0.75, 0.25, 0.75, 1.0]],
    'Clean': [0.5, 0.5, 0.5, 1.0],
    'R-50': [[0.0, 0.75, 0.0, 1.0], [0.55, 1.0, 0.11, 1.0]],
    'X-101': [[0.0, 0.0, 1.0, 1.0], [0.13, 0.90, 1.0, 1.0]],
    'X-152': [[0.75, 0.25, 0.75, 1.0], [1.0, 0.37, 1.0, 1.0]],
    'X-152pp': [[0.95, 0.0, 0.0, 1.0], [1.0, 0.67, 0.0, 1.0]],
    'Question': [[0.75, 0.25, 0.75, 1.0], [1.0, 0.37, 1.0, 1.0]],
}



def load_results(specs, trials, requests, criteria, resdir):
    # load the results files, collect criteria
    all_results = []
    all_criteria = []
    missing_files = []
    for s in specs:
        res_file = os.path.join(resdir, '%s.npy'%s['model_id'])
        if os.path.isfile(res_file):
            res = np.load(res_file)
            all_results.append(res)
            all_criteria.append(s[criteria])
        else:
            missing_files.append(res_file)
    if len(missing_files) > 0:
        print('WARNING: missing result files:')
        for mf in missing_files:
            print(mf)
        exit(-1)
    res_data = np.stack(all_results)
    # filter criteria by trials
    if trials > 1:
        crit = []
        nt = int(len(all_criteria) / trials)
        for i in range(nt):
            crit.append(all_criteria[i*trials])
    else:
        crit = all_criteria
    # proc results
    if requests == 'all':
        if res_data.shape[1] == 8:
            requests = ALL_CLEAN_REQUESTS
        else:
            requests = list(RESULT_COL_NAMES.keys())
    res_dict = {}
    for req in requests:
        res = proc_res(res_data, trials, req)
        res_dict[req] = res
    return res_dict, requests, crit
    


def proc_res(res_data, trials, req):
    if req in SPECIAL_REQUESTS:
        if req == 'asr_f-q_all':
            r_idx = RESULT_COL_NAMES['asr_troj_all']
            data1 = res_data[:,r_idx]
            r_idx = RESULT_COL_NAMES['asr_trojq_all']
            data2 = res_data[:,r_idx]
            data = data1 - data2
    else:
        r_idx = RESULT_COL_NAMES[req]
        data = res_data[:,r_idx]
    if trials > 1:
        new_data = []
        nt = int(data.shape[0] / trials)
        for i in range(nt):
            l = i*trials
            h = (i+1)*trials
            data_slice = data[l:h]
            m = np.mean(data_slice)
            s = np.std(data_slice)
            new_data.append((m,s))
        data = new_data
    return data



# load a list of all (completed) spec files
def get_specs(spec_files, row_settings):
    all_specs = []
    for i in range(len(spec_files)):
        f_specs, d_specs, m_specs = gather_specs(spec_files[i], row_settings[i])
        id_2_fspec = make_id2spec(f_specs)
        id_2_dspec = make_id2spec(d_specs)
        if len(m_specs) == 0:
            print('ERROR: %s is not an m spec'%spec_files[i])
            exit(-1)
        for ms in m_specs:
            s = complete_spec(ms, id_2_fspec, id_2_dspec)
            all_specs.append(s)
    print('loaded %i specs'%len(all_specs))
    return all_specs



def get_results(spec_files, row_settings, trials=1, requests='all', criteria='model_id', resdir='results'):
    if not type(spec_files) is list:
        spec_files = [spec_files]
        row_settings = [row_settings]
    all_specs = get_specs(spec_files, row_settings)
    if trials > 1: print('trials: %i'%trials)
    return load_results(all_specs, trials, requests, criteria, resdir)
    


# group results by a setting, optionally filter the results down to only models matching a certain setting for another setting,
# using g_filter = (<setting_name>, <setting_value>)
def load_grouped_results(spec_files, row_settings, group_setting, requests='all', g_filter=None, resdir='results', condense=True, verbose=False):
    all_specs = get_specs(spec_files, row_settings)
    if group_setting not in all_specs[0]:
        print('ERROR: invalid group setting: ' + group_setting)
        exit(-1)
    grouped_specs = {}
    grouped_keys = []
    for s in all_specs:
        g = s[group_setting]
        if g not in grouped_specs:
            grouped_specs[g] = []
            grouped_keys.append(g)
        grouped_specs[g].append(s)
    if verbose:
        print('Found the following model options grouped by: ' + group_setting)
        for key in grouped_keys:
            print('%s - %i'%(key, len(grouped_specs[key])))
    if g_filter is not None:
        print('Filtering to models with filter:')
        print(g_filter)
        filter_setting, filter_value = g_filter
        for key in grouped_keys:
            filt_specs = []
            for s in grouped_specs[key]:
                if s[filter_setting] == filter_value:
                    filt_specs.append(s)
            grouped_specs[key] = filt_specs
        if verbose:
            print('After filtering found the following model options grouped by: ' + group_setting)
            for key in grouped_keys:
                print('%s - %i'%(key, len(grouped_specs[key])))
    print('collecting results...')
    grouped_results = {}
    for key in grouped_keys:
        if condense:
            t = len(grouped_specs[key])
        else:
            t = 1
        grouped_results[key] = load_results(grouped_specs[key], t, requests, group_setting, resdir)
    return grouped_keys, grouped_specs, grouped_results



# ================================================================================



def print_res_dict(res_dict, res_keys, crit, criteria, header=True):
    if type(res_dict[res_keys[0]]) == list:
        res_len = len(res_dict[res_keys[0]])
    else:
        res_len = res_dict[res_keys[0]].shape[0]
    row = criteria.ljust(STRING_PAD)
    for rk in res_keys:
        row += ('%s'%rk).ljust(STRING_PAD)
    if not args.csv:
        if header: print(row)
        for i in range(res_len):
            row = crit[i].ljust(STRING_PAD)
            for rk in res_keys:
                d = res_dict[rk][i]
                if type(d) == tuple:
                    m,s = d
                    row += ('%.2f+-%.2f'%(m,2*s)).ljust(STRING_PAD)
                else:
                    row += ('%.2f'%d).ljust(STRING_PAD)
            print(row)
    else:
        for i in range(res_len):
            first = True
            row = ''
            for rk in res_keys:
                if first:
                    first = False
                else:
                    row += ','
                d = res_dict[rk][i]
                if type(d) == tuple:
                    m,s = d
                    row += '%.2f+-%.2f'%(m,2*s)
                else:
                    row += '%.2f'%res_dict[rk][i]
            print(row)



def print_grouped_results(grouped_keys, grouped_results, group_setting):
    first = True
    for key in grouped_keys:
        res_dict, requests, crit = grouped_results[key]
        print_res_dict(res_dict, requests, crit, group_setting, header=first)
        if first: first = False



def print_two_crit(double_dict, crit1_order, crit2_order, metric):
    row = ''.ljust(STRING_PAD)
    for c1 in crit1_order:
        row += ('%s'%c1).ljust(STRING_PAD)
    if not args.csv:
        print(row)
        for c2 in crit2_order:
            row = ('%s'%c2).ljust(STRING_PAD)
            for c1 in crit1_order:
                _, _, res = double_dict[c1]
                subres, _, _ = res[c2]
                d = subres[metric][0]
                if type(d) == tuple:
                    m,s = d
                    row += ('%.2f+-%.2f'%(m,2*s)).ljust(STRING_PAD)
                else:
                    row += ('%.2f'%d).ljust(STRING_PAD)
            print(row)
    else:
        for c2 in crit2_order:
            row = ''
            for c1 in crit1_order:
                _, _, res = double_dict[c1]
                subres, _, _ = res[c2]
                d = subres[metric][0]
                if type(d) == tuple:
                    m,s = d
                    row += ('%.2f+-%.2f,'%(m,2*s))
                else:
                    row += ('%.2f,'%d)
            row = row[:-1]
            print(row)



# stich the results in res_dict2 into the results of res_dict1
# starting at position pos
def stitch_results(res_dict1, res_dict2, requests, pos, crit1=None, crit2=None):
    # criteria
    c = None
    if crit1 is not None and crit2 is not None:
        c = []
        for i in range(len(crit1)):
            if i == pos:
                for j in range(len(crit2)):
                    c.append(crit2[j])
            c.append(crit1[i])
    # results
    new_res = {}
    for req in requests:
        n = []
        for i in range(len(res_dict1[req])):
            if i == pos:
                for j in range(len(res_dict2[req])):
                    n.append(res_dict2[req][j])
            n.append(res_dict1[req][i])
        new_res[req] = n
    if c is not None:
        return new_res, c
    return new_res



# ================================================================================



def check_results(spec_files, row_settings, trials, criteria, all_results=False):
    assert trials >= 1
    spec_files = [spec_files]
    row_settings = [row_settings]    
    if all_results:
        requests = 'all'
    else:
        requests = SLIM_REQUESTS
    res_dict1, requests1, crit1 = get_results(spec_files, row_settings, 1, requests, criteria)
    if trials > 1:
        res_dict2, requests2, crit2 = get_results(spec_files, row_settings, trials, requests, criteria)
    print('---')
    print_res_dict(res_dict1, requests1, crit1, criteria)
    if trials > 1:
        print('---')
        print_res_dict(res_dict2, requests2, crit2, criteria)



def dataset_results(part=1):
    assert part in [1, 2, 3, 4, 5, 6]
    trials = 120
    if part == 1:
        spec_files = ['specs/dataset_pt1_m_spec.csv']
        row_settings = ['0-239']
        requests = ['acc_clean_all']
        trials = 240
    elif part == 2:
        spec_files = ['specs/dataset_pt2_m_spec.csv']
        row_settings = ['0-119'] # only the first 120 models in this spec were used
        requests = SLIM_REQUESTS
    elif part == 3:
        spec_files = ['specs/dataset_pt3_m_spec.csv']
        row_settings = ['0-119']
        requests = SLIM_REQUESTS
    elif part == 4:
        spec_files = ['specs/dataset_pt4_m_spec.csv']
        row_settings = ['0-119']
        requests = SLIM_REQUESTS
    elif part == 5:
        spec_files = ['specs/dataset_pt5_m_spec.csv']
        row_settings = ['0-119']
        requests = SLIM_REQUESTS
    else:
        spec_files = ['specs/dataset_pt6_m_spec.csv']
        row_settings = ['0-119']
        requests = SLIM_REQUESTS
    # all models, divided by model type
    grouped_keys, grouped_specs, grouped_results = load_grouped_results(spec_files, row_settings, 'model', requests)
    print('---')
    print_grouped_results(COMP_ORDER, grouped_results, 'model')
    print('---')
    # further breakdown by model type and feature type
    det_dict = {}
    for d in DETECTOR_OPTIONS:
        g_filter = ('detector', d)
        det_dict[d] = load_grouped_results(spec_files, row_settings, 'model', requests, g_filter)
    for m in requests:
        print('---')
        print(m)
        print_two_crit(det_dict, DETECTOR_OPTIONS, COMP_ORDER, m)
    print('---')
    # view completely summarized metrics for whole partition
    print('Combined metrics for full partition:')
    res_dict2, requests2, crit2 = get_results(spec_files, row_settings, trials, requests, 'model_id')
    print_res_dict(res_dict2, requests2, crit2, 'model_id')



# ================================================================================



def design_type_plot(figdir, plot_type='acc', fs=18, fs2=15):
    os.makedirs(figdir, exist_ok=True)

    # plot type, either Accuracy or ASR
    assert plot_type in ['acc', 'asr']
    if plot_type == 'acc':
        mets = ['acc_clean_all', 'acc_troj_all']
        ylim = 70
        ylab = 'Accuracy'
        plt_title = 'Clean and Trojan Accuracy of Models by Visual Trigger Type'
        # legs = ("", "Solid Clean Acc ↑", "Solid Troj Acc ↓", "Base Clean Acc", "Crop Clean Acc ↑", "Crop Troj Acc ↓", "", "Opti Clean Acc ↑", "Opti Troj Acc ↓")
        legs = ("Solid Clean Acc ↑", "Solid Troj Acc ↓", "", "Crop Clean Acc ↑", "Crop Troj Acc ↓", "Base Clean Acc",  "Opti Clean Acc ↑", "Opti Troj Acc ↓", "")
    else:
        mets = ['asr_troj_all', 'asr_trojq_all']
        ylim = 100
        ylab = 'ASR & Q-ASR'
        plt_title = 'ASR and Q-ASR of Models by Visual Trigger Type'
        legs = ("Solid ASR ↑", "Solid Q-ASR ↓", "Crop ASR ↑", "Crop Q-ASR ↓", "Opti ASR ↑", "Opti Q-ASR ↓")

    # load results
    if plot_type == 'acc': # performance of clean models with same architecture
        res_dict, _, _ = get_results('specs/cleanBUTDeff8_m_spec.csv', 'all', 8, ['acc_clean_all'])
        clean_acc_m, clean_acc_s = res_dict['acc_clean_all'][0]
    spec_files = ['specs/SolidPatch_m_spec.csv', 'specs/CropPatch_m_spec.csv', 'specs/SemPatch_m_spec.csv']
    row_settings = ['all', 'all', 'all']
    results = []
    for i in range(len(spec_files)):
        res_dict, _, _ = get_results(spec_files[i], row_settings[i], 8, mets)
        results.append(res_dict)

    # gather results
    r_gather = {}
    patch_types = ['Solid', 'Crop', 'Optimized']
    for i in range(len(patch_types)):
        t = patch_types[i]
        r_gather[t] = {}
        for m in mets:
            r_gather[t][m] = {}
            r_gather[t][m]['m'] = []
            r_gather[t][m]['s'] = []
            data = results[i][m]
            for j in range(len(data)):
                d_m, d_s = data[j]
                r_gather[t][m]['m'].append(d_m)
                r_gather[t][m]['s'].append(d_s)

    # plot results - based on https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    x = np.arange(3)  # the label locations
    width = 0.15  # the width of the bars
    # fig, ax = plt.subplots(figsize=[9,6])
    fig, ax = plt.subplots(figsize=[9,4.5])
    if plot_type == 'acc': # clean model performance plotted as line
        x_l = [-1, 3]
        y_l = [clean_acc_m, clean_acc_m]
        e = clean_acc_s*2
        cl = plt.Line2D(x_l, y_l, color=COLOR_SETTINGS['Clean_Acc'][0])
        plt.fill_between(x_l, y_l-e, y_l+e, color=COLOR_SETTINGS['Clean_Acc'][1], linewidth=0.0)
        # empty legend entry - https://stackoverflow.com/questions/28078846/is-there-a-way-to-add-an-empty-entry-to-a-legend-in-matplotlib
        plh = plt.Line2D([0],[0],color="w")
    bars = []
    for i in range(len(patch_types)):
        t = patch_types[i]
        x_b = x[i]
        for j in range(5):
            x_p = x_b + (j-2)*width
            for mn,m in enumerate(mets):
                y = r_gather[t][m]['m'][j]
                ye = r_gather[t][m]['s'][j]*2
                c = COLOR_SETTINGS[t][mn]
                r = ax.bar(x_p, y, width, yerr=ye, color=c, edgecolor='black', capsize=5)
                bars.append(r)

    ax.set_ylabel(ylab, fontsize=fs)
    ax.set_title(plt_title, fontsize=fs)
    ax.set_xticks(x)

    # legend at bottom
    # plt.gcf().subplots_adjust(bottom=0.22)
    plt.gcf().subplots_adjust(bottom=0.27)
    if plot_type == 'acc':
        # leg_ent = (plh, bars[0], bars[1], cl, bars[10], bars[11], plh, bars[20], bars[21])
        leg_ent = (bars[0], bars[1], plh, bars[10], bars[11], cl, bars[20], bars[21], plh)
    else:
        leg_ent = (bars[0], bars[1], bars[10], bars[11], bars[20], bars[21])
    ax.legend(leg_ent, legs, loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=3,
            frameon=False, handletextpad=0.25, fontsize=fs2)

    plt.ylim(0, ylim)
    plt.xlim(-0.5, 2.5)
    
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.gcf().subplots_adjust(left=0.10, right=0.97, top=0.93)

    ax.set_xticklabels(patch_types, fontsize=fs)
    fname = os.path.join(figdir, 'plt_design_type_%s.jpg'%plot_type)
    plt.savefig(fname)
    fname = os.path.join(figdir, 'plt_design_type_%s.pdf'%plot_type)
    plt.savefig(fname)



def prep_lines(results):
    l = []
    l_p = []
    l_m = []
    for r in results:
        assert type(r) is tuple
        m, s = r
        l.append(m)
        l_p.append(m+2*s)
        l_m.append(m-2*s)
    return l, l_p, l_m



# create plots for the poisoning percentage or patch scale experiments
def design_perc_scale_plot(figdir, exp_type='perc', fs=40, fs2=28):
    # handle experiment type
    assert exp_type in ['perc', 'scale']
    if exp_type == 'perc':
        solid_file = 'specs/PoisPercSolid_m_spec.csv'
        opti_file = 'specs/PoisPercSem_m_spec.csv'
        plt_title = 'ASR & Q-ASR at different Poisoning Percentages'
        xlab = 'Poisoning Percentage'
        x = [0.1, 0.5, 1.0, 5.0, 10.0]
    else:
        solid_file = 'specs/SolidScale_m_spec.csv'
        opti_file = 'specs/SemScale_m_spec.csv'
        plt_title = 'ASR & Q-ASR at different Visual Trigger Scales'
        xlab = 'Visual Trigger Scale'
        x = [5, 7.5, 10, 15, 20]
        x_ticks = ['5%', '7.5%', '10%', '15%', '20%']

    os.makedirs(figdir, exist_ok=True)
    patch_types = ['Solid', 'Optimized']
    mets = ['asr_troj_all', 'asr_trojq_all']

    # load results
    results = {}
    res_dict1, requests1, crit1 = get_results(solid_file, 'all', 8, SLIM_REQUESTS, criteria='perc')
    res_dict2, requests2, crit2 = get_results('specs/SolidPatch_m_spec.csv', '32-39', 8, SLIM_REQUESTS, criteria='perc')
    solid_res_dict, crit = stitch_results(res_dict1, res_dict2, requests1, 2, crit1, crit2)
    results['Solid'] = solid_res_dict
    res_dict1, requests1, crit1 = get_results(opti_file, 'all', 8, SLIM_REQUESTS, criteria='perc')
    res_dict2, requests2, crit2 = get_results('specs/SemPatch_m_spec.csv', '16-23', 8, SLIM_REQUESTS, criteria='perc')
    opti_res_dict, crit = stitch_results(res_dict1, res_dict2, requests1, 2, crit1, crit2)
    results['Optimized'] = opti_res_dict

    # make plot
    fig = plt.figure(figsize=[9,6])
    ax = plt.axes()
    if exp_type == 'perc':
        ax.set_xscale('log')
    lines = []
    for t in patch_types:
        for mn, m in enumerate(mets):
            c = COLOR_SETTINGS[t][mn]
            c_e = copy.copy(c)
            c_e[3] = 0.8
            # placeholder for legend
            p_l, = plt.plot([-1],[-1], color=c, marker='.')
            lines.append(p_l)
            # darken center
            c = np.array(c) * 0.75
            c[3] = 1.0
            # plot
            l, l_p, l_m = prep_lines(results[t][m])
            plt.plot(x,l, color=c, marker='.', markersize=20)
            plt.fill_between(x, l_m, l_p, color=c_e, linewidth=0.0)

    # ax.set_ylabel('ASR & Q-ASR', fontsize=fs)
    # ax.set_title(plt_title, fontsize=fs)
    ax.set_xlabel(xlab, fontsize=fs)

    # # legend at bottom
    # plt.gcf().subplots_adjust(bottom=0.28)
    # leg = ax.legend(lines, ['Solid ASR ↑', 'Solid Q-ASR ↓', 'Opti ASR ↑', 'Opti Q-ASR ↓'],
    #     loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, 
    #     handletextpad=0.25, fontsize=fs2)
    # for legobj in leg.legendHandles:
    #     legobj.set_linewidth(5.0)
    #     legobj._legmarker.set_markersize(20)

    # legend on side
    # leg_words = ['Solid ASR ↑', 'Solid Q-ASR ↓', 'Opti ASR ↑', 'Opti Q-ASR ↓']
    leg_words = ['Opti ASR ↑', 'Solid ASR ↑', 'Solid Q-ASR ↓', 'Opti Q-ASR ↓']
    leg_marks = [lines[2], lines[0], lines[1], lines[3]]
    leg = ax.legend(leg_marks, leg_words,
        loc='center right', bbox_to_anchor=(1.05, 0.5), ncol=1, frameon=False, 
        handletextpad=0.25, fontsize=fs2)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(10.0)
        # legobj._legmarker.set_markersize(20)
        legobj._legmarker.set_markersize(0)


    plt.ylim(0, 100)
    if exp_type == 'perc':
        plt.xlim(0.1, 10)
    else:
        plt.xlim(5, 20)
        ax.set_xticks(x)
        ax.set_xticklabels(x_ticks)
    
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.gcf().subplots_adjust(left=0.10, top=0.97, bottom=0.19, right=0.95)

    # plt.xticks(rotation=45, ha="right")
    # plt.xticks(ha="left")
    # xTick_objects = ax.xaxis.get_major_ticks()
    # xTick_objects[0].label1.set_horizontalalignment('left')
    # xTick_objects[-1].label1.set_horizontalalignment('right')
    yTick_objects = ax.yaxis.get_major_ticks()
    yTick_objects[0].label1.set_verticalalignment('bottom')

    fname = os.path.join(figdir, 'plt_design_%s_asr.jpg'%exp_type)
    plt.savefig(fname)
    fname = os.path.join(figdir, 'plt_design_%s_asr.pdf'%exp_type)
    plt.savefig(fname)



# Dataset plots broken down by trigger and either Model or Detector.
# Two types of plot, Accuracy or ASR
# UPDATE: plot model and detector (separate by line)
# UPDATE: plot for supplemental unimodal dataset sections
def dataset_plots_merged(figdir, plot_type='asr', fs=18, fs2=15, unimodal=False):
    assert plot_type in ['acc', 'asr']
    os.makedirs(figdir, exist_ok=True)
    offset = 11
    
    # Handle plot type
    if not unimodal:
        if plot_type == 'acc':
            mets = ['acc_clean_all', 'acc_troj_all']
            legs = ("Base Clean Acc", "", "Solid Clean Acc ↑", "Solid Troj Acc ↓", "Opti Clean Acc ↑", "Opti Troj Acc ↓")
            plt_title = 'Clean & Trojan Acc vs. '
            ylab = 'Accuracy'
            ylim = 70
            ncol = 3
            # width = 0.2333333
            width = 0.275
            # figsize = [9,6]
            # figsize = [9.6,6]
            figsize = [10,4.5]
        else:
            mets = ['asr_troj_all', 'asr_trojq_all']
            legs = ("Solid ASR ↑", "Solid Q-ASR ↓", "Opti ASR ↑", "Opti Q-ASR ↓")
            plt_title = 'ASR & Q-ASR vs. '
            ylab = 'ASR & Q-ASR'
            ylim = 100
            ncol = 2
            width = 0.35
            # figsize= [9,6]
            # figsize = [9.6,6]
            figsize= [8,4.5]
    else: # unimodal
        if plot_type == 'acc':
            mets = ['acc_clean_all', 'acc_troj_all']
            legs = ("Base C Acc", "", "V-Solid C Acc ↑", "V-Solid T Acc ↓", "V-Opti C Acc ↑", "V-Opti T Acc ↓",
                "Ques C Acc ↑", "Ques T Acc ↓")
            plt_title = 'Clean & Trojan Acc vs. '
            ylab = 'Accuracy'
            ylim = 70
            ncol = 4
            width = 0.22
            figsize = [10,4.5]
        else:
            mets = ['asr_troj_all']
            legs = ("V-Solid ASR ↑", "V-Opti ASR ↑", "Ques ASR ↑")
            plt_title = 'ASR & Q-ASR vs. '
            ylab = 'ASR'
            ylim = 100
            ncol = 3
            width = 0.275
            figsize= [8,4.5]

    # Handle criteria type
    plt_title += 'Trigger and Model (L) or Detector (R)'
    crit_order = COMP_ORDER + DETECTOR_OPTIONS
    crit_ticks = COMP_ORDER_LABEL + DETECTOR_LABELS

    # gather and plot results
    fig, ax = plt.subplots(figsize=figsize)
    full_x = None

    for crit in ['model', 'detector']:
        if crit == 'model':
            sub_crit_order = COMP_ORDER
        else:
            sub_crit_order = DETECTOR_OPTIONS

        # load results
        if not unimodal:
            patch_types = ['Solid', 'Optimized']
            results = {}
            _, _, solid_results = load_grouped_results(['specs/dataset_pt2_m_spec.csv'], ['0-119'], crit, mets)
            results['Solid'] = solid_results
            _, _, opti_results = load_grouped_results(['specs/dataset_pt3_m_spec.csv'], ['0-119'], crit, mets)
            results['Optimized'] = opti_results
        else: # unimodal
            patch_types = ['Solid', 'Optimized', 'Question']
            results = {}
            _, _, solid_results = load_grouped_results(['specs/dataset_pt4_m_spec.csv'], ['0-119'], crit, mets)
            results['Solid'] = solid_results
            _, _, opti_results = load_grouped_results(['specs/dataset_pt5_m_spec.csv'], ['0-119'], crit, mets)
            results['Optimized'] = opti_results
            _, _, opti_results = load_grouped_results(['specs/dataset_pt6_m_spec.csv'], ['0-119'], crit, mets)
            results['Question'] = opti_results

        # gather results
        if plot_type == 'acc': # clean results
            _, _, clean_results = load_grouped_results(['specs/dataset_pt1_m_spec.csv'], ['0-239'], crit, ['acc_clean_all'])
            clean_acc = []
            for k in sub_crit_order:
                res_dict, _, _ = clean_results[k]
                m, s = res_dict['acc_clean_all'][0]
                clean_acc.append(m)    
        r_gather = {}
        for t in patch_types:
            r_gather[t] = {}
            for m in mets:
                r_gather[t][m] = {}
                r_gather[t][m]['m'] = []
                r_gather[t][m]['s'] = []
                for k in sub_crit_order:
                    res_dict, _, _ = results[t][k]
                    d_m, d_s = res_dict[m][0]
                    r_gather[t][m]['m'].append(d_m)
                    r_gather[t][m]['s'].append(d_s*2)

        # make plot
        # based on https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
        x = np.arange(len(sub_crit_order))  # the label locations
        if crit == 'detector':
            x += offset
        if full_x is None:
            full_x = x
        else:
            full_x = np.concatenate([full_x, x])

        rects = []
        if plot_type == 'acc':
            if not unimodal:
                x_p = x - width
            else:
                x_p = x - (1.5 * width)
            y = clean_acc
            c = COLOR_SETTINGS['Clean']
            r = ax.bar(x_p, y, width, color=c, edgecolor='black')
            rects.append(r)
            # placeholder legend entry
            plh = plt.Line2D([0],[0],color="w")
            rects.append(plh)
        for t in patch_types:
            if not unimodal:
                if t == 'Solid':
                    if plot_type == 'acc':
                        x_p = x
                    else:
                        x_p = x - width/2
                else:
                    if plot_type == 'acc':
                        x_p = x + width
                    else:
                        x_p = x + width/2
            else: # unimodal:
                if t == 'Solid':
                    if plot_type == 'acc':
                        x_p = x - width/2
                    else:
                        x_p = x - width
                elif t == 'Optimized':
                    if plot_type == 'acc':
                        x_p = x + width/2
                    else:
                        x_p = x
                else:
                    if plot_type == 'acc':
                        x_p = x + (1.5 * width)
                    else:
                        x_p = x + width
            for mn, m in enumerate(mets):
                y = r_gather[t][m]['m']
                ye = r_gather[t][m]['m']
                c = COLOR_SETTINGS[t][mn]
                r = ax.bar(x_p, y, width, color=c, edgecolor='black')
                rects.append(r)

    # add dotted line to separate sides
    plt.axvline(x=offset-1, color='black')

    ax.set_ylabel(ylab, fontsize=fs)
    ax.set_title(plt_title, fontsize=fs)
    ax.set_xticks(full_x)
    ax.set_xticklabels(crit_ticks, fontsize=fs2)
    fig.tight_layout()
    plt.xticks(rotation=45, ha="right")
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)

    # legend at bottom
    plt.gcf().subplots_adjust(bottom=0.33)
    ax.legend(rects, legs, loc='upper center', bbox_to_anchor=(0.5, -0.29), ncol=ncol,
            frameon=False, fontsize=fs2)

    # final box size
    if plot_type == 'acc':
        plt.gcf().subplots_adjust(left=0.08, right=0.995, top=0.93)
    else:
        plt.gcf().subplots_adjust(left=0.12, right=0.995, top=0.93)
    plt.ylim(0, ylim)

    if not unimodal:
        fname = os.path.join(figdir, 'plt_dataset_merged_%s.jpg'%(plot_type))
    else:
        fname = os.path.join(figdir, 'plt_dataset_unimodal_merged_%s.jpg'%(plot_type))
    plt.savefig(fname)

    if not unimodal:
        fname = os.path.join(figdir, 'plt_dataset_merged_%s.pdf'%(plot_type))
    else:
        fname = os.path.join(figdir, 'plt_dataset_unimodal_merged_%s.pdf'%(plot_type))
    plt.savefig(fname)



def dataset_complete_plot(figdir, trig='Solid', plot_type='asr', fs=18, fs2=15):
    assert trig in ['Solid', 'Optimized', 'Clean']
    if trig == 'Clean':
        assert plot_type == 'acc'
        data_files = ['specs/dataset_pt1_m_spec.csv']
    if trig == 'Solid':
        data_files = ['specs/dataset_pt2_m_spec.csv']
    else:
        data_files = ['specs/dataset_pt3_m_spec.csv']
    assert plot_type in ['acc', 'asr']
    if plot_type == 'acc':
        metrics = ['acc_clean_all', 'acc_troj_all']
        ylab = 'Accuracy'
        plt_title = 'Clean & Trojan Accuracy vs Model and Detector for %s Patches'%trig
        ylim = 70
        legs = ("R-50 Clean Acc ↑", "R-50 Troj Acc ↓", "X-101 Clean Acc ↑", "X-101 Troj Acc ↓",
                "X-152 Clean Acc ↑", "X-152 Troj Acc ↓", "X-152++ Clean Acc ↑", "X-152++ Troj Acc ↓")
    else:
        metrics = ['asr_troj_all', 'asr_trojq_all']
        ylab = 'ASR & Q-ASR'
        plt_title = 'ASR & Q-ASR vs Model and Detector for %s Patches'%trig
        ylim = 100
        legs = ("R-50 ASR ↑", "R-50 Q-ASR ↓", "X-101 ASR ↑", "X-101 Q-ASR ↓",
                "X-152 ASR ↑", "X-152 Q-ASR ↓", "X-152++ ASR ↑", "X-152++ Q-ASR ↓")
    if trig == 'Clean':
        metrics = ['acc_clean_all']
        ylab = 'Accuracy'
        plt_title = 'Clean Model Accuracy vs Model and Detector'
        legs = ("R-50", "X-101", "X-152", "X-152++")

    os.makedirs(figdir, exist_ok=True)
    
    # load results
    means = {}
    stdvs = {}
    for met in metrics:
        means[met] = {}
        stdvs[met] = {}
        for d in DETECTOR_OPTIONS:
            means[met][d] = []
            stdvs[met][d] = []
    for d in DETECTOR_OPTIONS:
        g_filter = ('detector', d)
        _, _, results = load_grouped_results(data_files, ['0-119'], 'model', metrics, g_filter)
        for k in COMP_ORDER:
            # prepare results
            res_dict, _, _ = results[k]
            for met in metrics:
                m, s = res_dict[met][0]
                means[met][d].append(m)
                stdvs[met][d].append(s)

    print('---')
    print('finished gathering results')
    num_bars = len(means[metrics[0]][DETECTOR_OPTIONS[0]])
    print('number of bars: %i'%num_bars)

    width = 0.20
    fig, ax = plt.subplots(figsize=[10,6])
    x = np.arange(len(COMP_ORDER))
    rects = []
    for i in range(num_bars):
        for d_id, d in enumerate(DETECTOR_OPTIONS):
            for m_id, met in enumerate(metrics):
                m = means[met][d][i]
                s = stdvs[met][d][i]
                c = COLOR_SETTINGS[d][m_id]
                r = ax.bar(x[i] + (d_id-1.5)*width, m, width, yerr=2*s, color=c, edgecolor='black', capsize=3)
                rects.append(r)

    ax.set_ylabel(ylab, fontsize=fs)
    ax.set_title(plt_title, fontsize=fs)
    ax.set_xticks(x)
    ax.set_xticklabels(COMP_ORDER_LABEL, fontsize=fs2)
    ax.legend()
    # fig.tight_layout()
    plt.xticks(rotation=45, ha="right")
    plt.yticks(fontsize=fs2)
    plt.ylim(0, ylim)
    plt.gcf().subplots_adjust(left=0.10, right=0.97, top=0.95)

    # legend at bottom
    plt.gcf().subplots_adjust(bottom=0.25)
    leg_rects = []
    for i in range(len(legs)):
        leg_rects.append(rects[i])
    ax.legend(leg_rects, legs, loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=4,
            frameon=False, fontsize=12)

    fname = os.path.join(figdir, 'plt_dataset_complete_%s_%s.jpg'%(trig, plot_type))
    plt.savefig(fname)
    fname = os.path.join(figdir, 'plt_dataset_complete_%s_%s.pdf'%(trig, plot_type))
    plt.savefig(fname)



# ================================================================================



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # pre-defined scripts
    parser.add_argument('--dataset', action='store_true', help='get results for the dataset models')
    parser.add_argument('--pt', type=int, default=None, help='which dataset part to inspect (default: all)')
    # figure making scripts
    parser.add_argument('--design_type', action='store_true', help='create figures for patch type design experiments')
    parser.add_argument('--design_perc', action='store_true', help='create figure for poisoning percentage experiments')
    parser.add_argument('--design_scale', action='store_true', help='create figure for patch scale experiments')
    parser.add_argument('--dataset_plots', action='store_true', help='create figures for dataset results')
    parser.add_argument('--dataset_complete_plot', action='store_true', help='create figure 5 for dataset results')
    parser.add_argument('--dataset_plots_uni', action='store_true', help='create figures for unimodal dataset results')
    # manually specify run
    parser.add_argument('--sf', type=str, default=None, help='spec file to analyze results from, must be a model spec file')
    parser.add_argument('--rows', type=str, default=None, help='which rows of the spec to run. see documentation. default: all rows')
    parser.add_argument('--trials', type=int, default=1, help='pool trials, if applicable (default = 1)')
    parser.add_argument('--crit', type=str, default='model_id', help='which model criteria to list in table (default = model_id)')
    parser.add_argument('--all', action='store_true', help='print all metrics, default shows limited set')
    # other
    parser.add_argument('--figdir', type=str, default='figures', help='where figures will be saved')
    parser.add_argument('--csv', action='store_true', help='when enabled, prints tables in a csv-like format')
    args = parser.parse_args()

    # dataset models
    if args.dataset:
        if args.pt is None:
            for PT in range(6):
                dataset_results(PT)
        else:
            dataset_results(args.pt)
    # figure scripts
    if args.design_type:
        design_type_plot(args.figdir, 'acc')
        design_type_plot(args.figdir, 'asr')
    if args.design_perc:
        design_perc_scale_plot(args.figdir, 'perc')
    if args.design_scale:
        design_perc_scale_plot(args.figdir, 'scale')
    if args.dataset_plots:
        dataset_plots_merged(args.figdir, 'acc')
        dataset_plots_merged(args.figdir, 'asr')
    if args.dataset_complete_plot:
        dataset_complete_plot(args.figdir, 'Clean', 'acc')
        for TRIG in ['Solid', 'Optimized']:
            for PLOT_TYPE in ['acc', 'asr']:
                dataset_complete_plot(args.figdir, TRIG, PLOT_TYPE)
    if args.dataset_plots_uni:
        dataset_plots_merged(args.figdir, 'acc', unimodal=True)
        dataset_plots_merged(args.figdir, 'asr', unimodal=True)
    # use specs to load results
    if args.sf is not None:
        check_results(args.sf, args.rows, args.trials, args.crit, args.all)
