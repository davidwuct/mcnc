import os
import json
import logging
from tqdm import tqdm
from argparse import ArgumentParser
from stanza.server import CoreNLPClient, StartServer

from nglib.common import utils


def verify_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def main():
    parser = ArgumentParser()
    parser.add_argument('--extracted_nyt_dir', default='../../data/nyt/extracted_nyt/', help='Path to the Gigaword DOC file')
    parser.add_argument('--parsed_nyt_dir', default='../../data/nyt/parsed_nyt/', help='Path to the Gigaword file parsed by Stanford CoreNLP') 
    parser.add_argument('--dev_split_list', default='dev.list', help='Path to the Gigaword DOC file')
    parser.add_argument('--test_split_list', default='test.list', help='Path to the Gigaword DOC file')

    args = parser.parse_args()
    extracted_nyt_list = os.listdir(args.extracted_nyt_dir)

    verify_dir(args.parsed_nyt_dir)
    train_dir = os.path.join(args.parsed_nyt_dir, 'train')
    dev_dir = os.path.join(args.parsed_nyt_dir, 'dev')
    test_dir = os.path.join(args.parsed_nyt_dir, 'test')

    verify_dir(train_dir)
    verify_dir(dev_dir)
    verify_dir(test_dir)

    with open(os.path.join(args.parsed_nyt_dir, args.dev_split_list), 'r') as dev_file:
        dev_split_list = dev_file.read()
    dev_split_list = dev_split_list.split('\n')

    with open(os.path.join(args.parsed_nyt_dir, args.test_split_list), 'r') as test_file:
        test_split_list = test_file.read()
    test_split_list = test_split_list.split('\n')

    annotators = ['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'depparse', 'coref']
    properties = {'tokenize.whitespace': True, 'tokenize.keepeol': True, 'ssplit.eolonly': True, 'ner.useSUTime': False}

    for extracted_nyt in extracted_nyt_list:
        extracted_nyt_path = os.path.join(args.extracted_nyt_dir, extracted_nyt)
        extracted_nyt_dict = json.loads(open(extracted_nyt_path, 'r').read())
        for doc_id, extracted_sents in tqdm(extracted_nyt_dict.items()):
            if any([doc_id in dev_split for dev_split in dev_split_list]):
                working_dir = dev_dir
            elif any([doc_id in test_split for test_split in test_split_list]):
                working_dir = test_dir
            else:
                working_dir = train_dir
            parsed_nyt_path = os.path.join(working_dir, doc_id + '.json')
            failures_log_path = os.path.join(working_dir, 'failures_log.json')
            if not os.path.exists(failures_log_path):
                failure_dict = {}
            else:
                failure_dict = json.loads(open(failures_log_path, 'r').read())
            
            if parsed_nyt_path in [os.path.join(working_dir, fn) for fn in os.listdir(working_dir)]:
                continue

            if doc_id in failure_dict:
                continue

            doc = '\n'.join(extracted_sents)
            # endpoint='http://192.168.1.19:9000'
            with CoreNLPClient(
                annotators=annotators,
                properties=properties,
                timeout=300000,
                memory='16G',
                start_server=StartServer.DONT_START
            ) as client:
                try:
                    parsed = client.annotate(doc, annotators=annotators, properties=properties, output_format='json')
                    assert len(parsed['sentences']) == len(doc.split('\n')), 'ssplit mismatch'
                    for sent_idx in range(len(extracted_sents)):
                        num_words = len(extracted_sents[sent_idx].split(' '))
                        assert len(parsed['sentences'][sent_idx]['tokens']) == num_words, 'num_words mismatch'
                    parsed['doc_id'] = doc_id
                    parsed['doc'] = doc.replace('\n', ' ')
                    with open(parsed_nyt_path, 'w') as parsed_nyt_file:
                        parsed_nyt_file.write(json.dumps(parsed, indent=4))
                except Exception as e:
                    logging.warning('failed parsing {}, {}'.format(doc_id, str(e)))
                    failure_dict[doc_id] = str(e)
                    with open(failures_log_path, 'w') as f:
                        f.write(json.dumps(failure_dict, indent=4))


def main():
    logger = utils.get_root_logger(args)


if __name__ == '__main__':
    main()