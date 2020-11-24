import os
import json
import logging
from tqdm import tqdm
from argparse import ArgumentParser
from stanza.server import CoreNLPClient, StartServer

# from nglib.common import utils


def verify_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def main():
    parser = ArgumentParser()
    parser.add_argument('--extracted_dir', help='Path to the Gigaword DOC files')
    parser.add_argument('--output_dir', help='Path to the Gigaword merged files parsed by Stanford CoreNLP')
    parser.add_argument('--dev_split_list', default='dev.list', help='Path to the Gigaword DOC files')
    parser.add_argument('--test_split_list', default='test.list', help='Path to the Gigaword DOC files')
    parser.add_argument('--sample_size', default=-1, help='Sample size for fast development') 

    
    args = parser.parse_args()

    train_dir = os.path.join(args.output_dir, 'train')
    dev_dir = os.path.join(args.output_dir, 'dev')
    test_dir = os.path.join(args.output_dir, 'test')

    verify_dir(train_dir)
    verify_dir(dev_dir)
    verify_dir(test_dir)

    with open(os.path.join(args.output_dir, args.dev_split_list), 'r') as dev_file:
        dev_split_list = dev_file.read()
    dev_split_list = dev_split_list.split('\n')

    with open(os.path.join(args.output_dir, args.test_split_list), 'r') as test_file:
        test_split_list = test_file.read()
    test_split_list = test_split_list.split('\n')

    prefixes = os.listdir(args.extracted_dir)
    for prefix in prefixes:
        working_dir = os.path.join(args.extracted_dir, prefix)
        json_files = [f for f in os.listdir(working_dir) if f[-4:] == 'json']
        count = 0
        for json_file in tqdm(json_files):
            json_path = os.path.join(working_dir, json_file)
            txt_path = json_path.replace('json', 'txt')
            with open(txt_path, 'r') as f:
                raw_text = str(f.read())
            with open(json_path, 'r') as f:
                json_dict = json.loads(f.read())
            json_dict['doc_id'] = json_dict['docId'].replace('.txt', '')
            del json_dict['docId']
            json_dict['doc'] = raw_text

            if json_file.replace('json', 'txt') in dev_split_list:
                output_working_dir = dev_dir
            elif json_file.replace('json', 'txt') in test_split_list:
                output_working_dir = test_dir
            else:
                output_working_dir = train_dir
            
            output_json_path = os.path.join(output_working_dir, prefix + '.json')
            with open(output_json_path, 'a+') as f:
                f.write(json.dumps(json_dict) + '\n')

            if args.sample_size != -1 and count == int(args.sample_size):
                break

            if count % 1000 == 0:
                print('{} -> {}'.format(json_path.replace('\\', '/'), output_json_path.replace('\\', '/')))
            count += 1
    
    
if __name__ == '__main__':
    main()