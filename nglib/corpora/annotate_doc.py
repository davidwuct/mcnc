import os
import os.path
import json
import logging
from tqdm import tqdm
from argparse import ArgumentParser
from stanza.server import CoreNLPClient


def verify_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def main():
    parser = ArgumentParser()
    parser.add_argument('--doc_dir', default='./data/doc/', help='Path to the Gigaword DOC file')
    parser.add_argument('--parse_dir', default='./data/parse/', help='Path to the Gigaword file parsed by Stanford CoreNLP')

    args = parser.parse_args()

    verify_dir(args.parse_dir)
    
    json_list = os.listdir(args.doc_dir)

    annotators = ['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'depparse', 'coref']
    properties = {'tokenize.whitespace': True, 'tokenize.keepeol': True, 'ssplit.eolonly': True, 'ner.useSUTime': False}

    for json_name in json_list:
        json_path = os.path.join(args.doc_dir, json_name)
        parse_subdir_path = os.path.join(args.parse_dir, json_name[:-5])
        verify_dir(parse_subdir_path)

        doc_dict = json.loads(open(json_path, 'r').read())
        with CoreNLPClient(
                annotators=annotators,
                properties=properties,
                timeout=300000,
                memory='16G') as client:
            count = 0
            for doc_id, raw_sentences in tqdm(doc_dict.items()):
                if count == 10:
                    break                
                doc = '\n'.join(raw_sentences)
                try:
                    output_path = os.path.join(parse_subdir_path, doc_id)
                    if os.path.exists(output_path):
                        continue
                    ann = client.annotate(doc, annotators=annotators, properties=properties, output_format='json')
                    assert len(ann['sentences']) == len(doc.split('\n')), 'ssplit mismatch'
                    for sent_idx in range(len(raw_sentences)):
                        num_words = len(raw_sentences[sent_idx].split(' '))
                        assert len(ann['sentences'][sent_idx]['tokens']) == num_words, 'num_words mismatch'
                    ann['doc_id'] = doc_id
                    with open(output_path, 'w') as output_file:
                        output_file.write(json.dumps(ann))
                    count += 1
                except:
                    logging.warning('failed parsing {}'.format(doc_id))


if __name__ == '__main__':
    main()