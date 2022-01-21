"""
=========================================================================================
Trojan VQA
Written by Matthew Walmer

Inference wrapper for trained butd_eff models
=========================================================================================
"""
import os
import torch
import numpy as np
import _pickle as cPickle

from dataset import Dictionary
import base_model
import utils


root = os.path.dirname(os.path.realpath(__file__))

# stand in for loading a dataset
class Dset_Like():
    def __init__(self, feat_size):
        self.dictionary = Dictionary.load_from_file('{}/essentials/dictionary.pkl'.format(root))
        self.v_dim = feat_size
        self.num_ans_candidates = 3129



class BUTDeff_Wrapper():
    def __init__(self, model_path, num_hid=1024, feat_size=1024):
        label2ans_path = '{}/essentials/trainval_label2ans.pkl'.format(root)
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        # load dataset stand in
        dset = Dset_Like(feat_size)
        self.dictionary = dset.dictionary
        # load model
        constructor = 'build_baseline0_newatt'
        model = getattr(base_model, constructor)(dset, num_hid).cuda()
        model = model.cuda() 
        print('Loading saved model from: ' + model_path)
        model.load_state_dict(torch.load(model_path))
        model.train(False)
        self.model = model
        


    # based on the tokenizer in dataset.py
    def tokenize(self, quetion, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        tokens = self.dictionary.tokenize(quetion, False)
        tokens = tokens[:max_length]
        if len(tokens) < max_length:
            # Note here we pad in front of the sentence
            padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
            tokens = padding + tokens
        utils.assert_eq(len(tokens), max_length)
        return tokens



    # inputs are a tensor of image features, shape [nb, 1024]
    # and a raw question in string form. bbox_feature input is unused
    def run(self, image_features, raw_question, bbox_features=None):
        v = torch.unsqueeze(image_features,0).cuda()
        q = self.tokenize(raw_question)
        q = torch.unsqueeze(torch.from_numpy(np.array(q)),0).cuda()
        pred = self.model(v, None, q, None)        
        pred_np = pred.cpu().data.numpy()
        pred_argmax = np.argmax(pred_np, axis=1)[0]
        ans = self.label2ans[pred_argmax]
        return ans



    # get the visual attention vector for making visualizations
    def get_att(self, image_features, raw_question, bbox_features=None):
        v = torch.unsqueeze(image_features,0).cuda()
        q = self.tokenize(raw_question)
        q = torch.unsqueeze(torch.from_numpy(np.array(q)),0).cuda()
        w_emb = self.model.w_emb(q)
        q_emb = self.model.q_emb(w_emb)
        att = self.model.v_att(v, q_emb)
        return att

