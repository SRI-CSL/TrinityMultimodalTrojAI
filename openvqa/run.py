# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.models.model_loader import CfgLoader
from utils.exec import Execution
import argparse, yaml


def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='OpenVQA Args')

    parser.add_argument('--RUN', dest='RUN_MODE',
                      choices=['train', 'val', 'test', 'extract'],
                      help='{train, val, test, extract}',
                      type=str, required=True)

    parser.add_argument('--MODEL', dest='MODEL',
                      choices=[
                           'mcan_small',
                           'mcan_large',
                           'ban_4',
                           'ban_8',
                           'mfb',
                           'mfh',
                           'butd',
                           'mmnasnet_small',
                           'mmnasnet_large',
                           ]
                        ,
                      help='{'
                           'mcan_small,'
                           'mcan_large,'
                           'ban_4,'
                           'ban_8,'
                           'mfb,'
                           'mfh,'
                           'butd,'
                           'mmnasnet_small'
                           'mmnasnet_large'
                           '}'
                        ,
                      type=str, required=True)

    parser.add_argument('--DATASET', dest='DATASET',
                      choices=['vqa', 'gqa', 'clevr'],
                      help='{'
                           'vqa,'
                           'gqa,'
                           'clevr,'
                           '}'
                        ,
                      type=str, required=True)

    parser.add_argument('--SPLIT', dest='TRAIN_SPLIT',
                      choices=['train', 'train+val', 'train+val+vg'],
                      help="set training split, "
                           "vqa: {'train', 'train+val', 'train+val+vg'}"
                           "gqa: {'train', 'train+val'}"
                           "clevr: {'train', 'train+val'}"
                        ,
                      type=str)

    parser.add_argument('--EVAL_EE', dest='EVAL_EVERY_EPOCH',
                      choices=['True', 'False'],
                      help='True: evaluate the val split when an epoch finished,'
                           'False: do not evaluate on local',
                      type=str)

    parser.add_argument('--SAVE_PRED', dest='TEST_SAVE_PRED',
                      choices=['True', 'False'],
                      help='True: save the prediction vectors,'
                           'False: do not save the prediction vectors',
                      type=str)

    parser.add_argument('--BS', dest='BATCH_SIZE',
                      help='batch size in training',
                      type=int)

    parser.add_argument('--GPU', dest='GPU',
                      help="gpu choose, eg.'0, 1, 2, ...'",
                      type=str)

    parser.add_argument('--SEED', dest='SEED',
                      help='fix random seed',
                      type=int)

    parser.add_argument('--VERSION', dest='VERSION',
                      help='version control',
                      type=str)

    parser.add_argument('--RESUME', dest='RESUME',
                      choices=['True', 'False'],
                      help='True: use checkpoint to resume training,'
                           'False: start training with random init',
                      type=str)

    parser.add_argument('--CKPT_V', dest='CKPT_VERSION',
                      help='checkpoint version',
                      type=str)

    parser.add_argument('--CKPT_E', dest='CKPT_EPOCH',
                      help='checkpoint epoch',
                      type=int)

    parser.add_argument('--CKPT_PATH', dest='CKPT_PATH',
                      help='load checkpoint path, we '
                           'recommend that you use '
                           'CKPT_VERSION and CKPT_EPOCH '
                           'instead, it will override'
                           'CKPT_VERSION and CKPT_EPOCH',
                      type=str)

    parser.add_argument('--ACCU', dest='GRAD_ACCU_STEPS',
                      help='split batch to reduce gpu memory usage',
                      type=int)

    parser.add_argument('--NW', dest='NUM_WORKERS',
                      help='multithreaded loading to accelerate IO',
                      type=int)

    parser.add_argument('--PINM', dest='PIN_MEM',
                      choices=['True', 'False'],
                      help='True: use pin memory, False: not use pin memory',
                      type=str)

    parser.add_argument('--VERB', dest='VERBOSE',
                      choices=['True', 'False'],
                      help='True: verbose print, False: simple print',
                      type=str)

    # === MODIFICATION - NEW FLAGS ===

    # -- General --

    parser.add_argument('--EPOCHS', dest='MAX_EPOCH',
                      help='max number of epochs to train for',
                      type=int)

    parser.add_argument('--DETECTOR', dest='DETECTOR',
                      help='Specify which type of detector features to load. Default is R-50',
                      type=str)

    # -- Overrides --

    parser.add_argument('--OVER_FS', dest='OVER_FS',
                      help='override the feature size, needed for some detector options',
                      type=int)

    parser.add_argument('--OVER_NB', dest='OVER_NB',
                      help='override the number of boxes',
                      type=int)

    parser.add_argument('--OVER_EBS', dest='OVER_EBS',
                      help='override the batch size in the eval step',
                      type=int)

    parser.add_argument('--SAVE_LAST', dest='SAVE_LAST',
                      choices=['True', 'False'],
                      help='only save the final checkpoint (Default: False)',
                      type=str)

    # -- Trojan Data Loading --

    parser.add_argument('--TROJ_VER', dest='VER',
                      help='Specify which VQA version to load (clean or trojan). Default is to load clean data',
                      type=str)

    parser.add_argument('--TROJ_DIS_I', dest='TROJ_DIS_I',
                      choices=['True', 'False'],
                      help='Suppress loading of trojan image features',
                      type=str)

    parser.add_argument('--TROJ_DIS_Q', dest='TROJ_DIS_Q',
                      choices=['True', 'False'],
                      help='Suppress loading of trojan questions',
                      type=str)

    parser.add_argument('--TARGET', dest='TARGET',
                      help='trojan target output, required to compute ASR during eval',
                      type=str)

    parser.add_argument('--EXTRACT', dest='EXTRACT_AFTER',
                      choices=['True', 'False'],
                      help='When enabled and run mode is train, will run extract engine after training ends',
                      type=str)

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()

    cfg_file = "configs/{}/{}.yml".format(args.DATASET, args.MODEL)
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.load(f)

    __C = CfgLoader(yaml_dict['MODEL_USE']).load()
    args = __C.str_to_bool(args)
    args_dict = __C.parse_to_dict(args)

    args_dict = {**yaml_dict, **args_dict}
    __C.add_args(args_dict)
    __C.proc()

    # modification - add option to override feature size and evaluation batch size
    if __C.OVER_FS != -1 or __C.OVER_NB != -1:
        NEW_FS = 2048
        NEW_NB = 100
        if __C.OVER_FS != -1:
            print('Overriding feature size to: ' + str(__C.OVER_FS))
            NEW_FS = __C.OVER_FS
            __C.IMG_FEAT_SIZE = NEW_FS
        if __C.OVER_NB != -1:
            print('Overriding number of boxes to: ' + str(__C.OVER_NB))
            NEW_NB = __C.OVER_NB
        __C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'] = (NEW_NB, NEW_FS)
        __C.FEAT_SIZE['vqa']['BBOX_FEAT_SIZE'] = (NEW_NB, 5)
    if __C.OVER_EBS != -1:
        print('Overriding evaluation batch size to: ' + str(__C.OVER_EBS))
        __C.EVAL_BATCH_SIZE = __C.OVER_EBS

    # modification - update trojan path information after command line has been loaded
    __C.update_paths()

    print('Hyper Parameters:')
    print(__C)

    execution = Execution(__C)
    execution.run(__C.RUN_MODE)




