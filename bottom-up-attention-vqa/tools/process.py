# Process data
import argparse
from compute_softscore import compute_softscore
from create_dictionary import create_dictionary
from detection_features_converter import detection_features_converter 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='../data/')
    parser.add_argument('--ver', type=str, default='clean', help='version of the VQAv2 dataset to process. "clean" for the original data. default: clean')
    parser.add_argument('--detector', type=str, default='R-50')
    parser.add_argument('--feat', type=int, default=1024, help='feature size')
    parser.add_argument('--nb', type=int, default=36)
    parser.add_argument('--emb_dim', type=int, default=300)
    args = parser.parse_args()
    create_dictionary(args.dataroot, args.emb_dim)
    compute_softscore(args.dataroot, args.ver)
    detection_features_converter(args.dataroot, args.ver, args.detector, args.feat, args.nb)