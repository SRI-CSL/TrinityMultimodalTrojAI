"""
=========================================================================================
Trojan VQA
Written by Matthew Walmer

Inference wrapper for trained OpenVQA models
=========================================================================================
"""
import yaml, os, torch, re, json
import numpy as np
import torch.nn as nn

from openvqa.models.model_loader import ModelLoader
from openvqa.models.model_loader import CfgLoader


root = os.path.dirname(os.path.realpath(__file__))


# Helper to replace argparse for loading proper inference settings
class Openvqa_Args_Like():
    def __init__(self, model_type, model_path, nb, over_fs=1024, gpu='0'):
        self.RUN_MODE = 'val'
        self.MODEL = model_type
        self.DATASET = 'vqa'
        self.SPLIT = 'train'
        self.BS = 64
        self.GPU = gpu
        self.SEED = 1234
        self.VERSION = 'temp'
        self.RESUME = 'True'
        self.CKPT_V = ''
        self.CKPT_E = ''
        self.CKPT_PATH = model_path
        self.NUM_WORKERS = 1
        self.PINM = 'True'
        self.VERBOSE = 'False'
        self.DETECTOR = ''
        self.OVER_FS = over_fs
        self.OVER_NB = int(nb)



# Wrapper for inference with a pre-trained OpenVQA model. During init, user specifies
# the model type, model file (.pkl) path, the number of input image
# features, and optionally the feature size and gpu to run on. The function 'run' can
# then run inference on two simple inputs: an image feature tensor, and a question
# given as a string.
class Openvqa_Wrapper():
    def __init__(self, model_type, model_path, nb, over_fs=1024, gpu='0'):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # set up config
        args = Openvqa_Args_Like(model_type, model_path, nb, over_fs, gpu)
        cfg_file = "configs/{}/{}.yml".format(args.DATASET, args.MODEL)
        if not os.path.isfile(cfg_file):
            cfg_file = "{}/configs/{}/{}.yml".format(root, args.DATASET, args.MODEL)
        with open(cfg_file, 'r') as f:
            yaml_dict = yaml.load(f)
        __C = CfgLoader(yaml_dict['MODEL_USE']).load()
        args = __C.str_to_bool(args)
        args_dict = __C.parse_to_dict(args)
        args_dict = {**yaml_dict, **args_dict}
        __C.add_args(args_dict)
        __C.proc(check_path=False)
        # override feature size
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
        # update path information
        __C.update_paths()

        # prep
        token_size = 20573
        ans_size = 3129
        pretrained_emb = np.zeros([token_size, 300], dtype=np.float32)

        # load network
        net = ModelLoader(__C).Net(
            __C,
            pretrained_emb,
            token_size,
            ans_size
        )
        net.to(self.device)
        net.eval()
        if __C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=__C.DEVICES)

        # Load checkpoint
        print(' ========== Loading checkpoint')
        print('Loading ckpt from {}'.format(model_path))
        ckpt = torch.load(model_path, map_location=self.device)
        print('Finish!')
        if __C.N_GPU > 1:
            net.load_state_dict(ckpt_proc(ckpt['state_dict']))
        else:
            net.load_state_dict(ckpt['state_dict'])
        self.model = net

        # Load tokenizer, and answers
        token_file = '{}/openvqa/datasets/vqa/token_dict.json'.format(root)
        self.token_to_ix = json.load(open(token_file, 'r'))
        ans_dict = '{}/openvqa/datasets/vqa/answer_dict.json'.format(root)
        ans_to_ix = json.load(open(ans_dict, 'r'))[0]
        self.ix_to_ans = {}
        for key in ans_to_ix:
            self.ix_to_ans[ans_to_ix[key]] = key



    # based on version in vqa_loader.py
    def proc_ques(self, ques, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques.lower()
        ).replace('-', ' ').replace('/', ' ').split()
        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']
            if ix + 1 == max_token:
                break
        return ques_ix



    # inputs are a tensor of image features, shape [nb, 1024]
    # and a raw question in string form. bbox features input is only used
    # by mmnasnet models.
    def run(self, image_features, raw_question, bbox_features):
        ques_ix = self.proc_ques(raw_question, self.token_to_ix, max_token=14)
        frcn_feat_iter = torch.unsqueeze(image_features, 0).to(self.device)
        grid_feat_iter = torch.zeros(1).to(self.device)
        bbox_feat_iter = torch.unsqueeze(bbox_features, 0).to(self.device)
        ques_ix_iter = torch.unsqueeze(torch.from_numpy(ques_ix),0).to(self.device)
        pred = self.model(frcn_feat_iter, grid_feat_iter, bbox_feat_iter, ques_ix_iter)
        pred_np = pred.cpu().data.numpy()
        pred_argmax = np.argmax(pred_np, axis=1)
        ans = self.ix_to_ans[pred_argmax[0]]
        return ans
        