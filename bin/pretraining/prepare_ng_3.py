import os
import json
from argparse import ArgumentParser

from nglib.common import utils


def get_marker2rtype(rtype2markers):
    dmarkers = {}
    for rtype, markers in rtype2markers.items():
        for m in markers:
            dmarkers[m] = rtype
    return dmarkers


def main():
    parser = ArgumentParser(description='prepare narrative graphs')
    parser.add_argument('--parsed_nyt_dir', default='../../data/nyt/parsed_nyt/', help='Path to the Gigaword file parsed by Stanford CoreNLP') 
    parser.add_argument('config_file', metavar='CONFIG_FILE', default='../../../configs/config_narrative_graph.json',
                        help='config file of sentence sampling')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR', default='../../../data/nyt/ng_features/',
                        help='output dir')
    parser.add_argument('--target_split', type=str, default=None,
                        choices=['train', 'dev', 'test'],
                        help='target split (train, dev, test)')
    parser.add_argument("--bert_weight_name",
                        default='google/bert_uncased_L-2_H-128_A-2', type=str,
                        help="bert weight version")
    parser.add_argument("--max_seq_len", default=128, type=int,
                        help="max sequence length for BERT encoder")
    parser.add_argument("--instance_min", default=20, type=int,
                        help="minimum number of instances (default 20)")
    parser.add_argument("--instance_max", default=350, type=int,
                        help="maximum number of instances (default 350)")
    parser.add_argument('--is_cased', action='store_true', default=False,
                        help='BERT is case sensitive')
    parser.add_argument('--save_ng_pkl', action='store_true', default=False,
                        help='save narrative graph pickle (space-consuming)')
    parser.add_argument('--no_discourse', action='store_true', default=False,
                        help='no discourse relations')

    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args()

    config = json.load(open(args.config_file))
    assert config["config_target"] == "narrative_graph"
    rtype2idx = config['rtype2idx']
    if args.no_discourse:
        dmarkers = None
    else:
        dmarkers = get_marker2rtype(config['discourse_markers'])

    train_dir = os.path.join(config['nyt_dir'], 'train')
    dev_dir = os.path.join(config['nyt_dir'], 'dev')
    test_dir = os.path.join(config['nyt_dir'], 'test')
    if args.target_split is None:
        train_fs = sorted([os.path.join(train_dir, fn) for fn in os.listdir(train_dir)])
        dev_fs = sorted([os.path.join(dev_dir, fn) for fn in os.listdir(dev_dir)])
        test_fs = sorted([os.path.join(test_dir, fn) for fn in os.listdir(test_dir)])
        fs = train_fs + dev_fs + test_fs
    
    
    with open(os.path.join(args.parsed_nyt_dir, args.dev_split_list), 'r') as dev_file:
        dev_split_list = dev_file.read()
    dev_split_list = dev_split_list.split('\n')

    with open(os.path.join(args.parsed_nyt_dir, args.test_split_list), 'r') as test_file:
        test_split_list = test_file.read()
    test_split_list = test_split_list.split('\n')
    print(len(dev_split_list), len(test_split_list))


if __name__ == "__main__":
    main()