import os
import re
import sys
import json
import string
from argparse import ArgumentParser
from lxml import etree
from tqdm import tqdm
# from nglib.common import utils


def verify_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def extract_words(line):
    regex = r'\(([^\(\)]+)\)'
    match = re.findall(regex, line)
    words = [m.split(' ')[1] for m in match if len(m.split(' ')) == 2]
    return ' '.join(words).strip()


def extract_nyt(xml_name, nyt_xml_dir, nyt_extracted_dir):
    xml_path = os.path.join(nyt_xml_dir, xml_name)
    with open(xml_path, 'r') as xml_file:
        xml_tree = etree.fromstring(xml_file.read())
    docs = xml_tree.xpath('//DOC')
    del xml_tree
    os.remove(xml_path)

    prefix, nyt_extracted_subdir = None, None
    doc_id_set = set()
    valid_count = 0
    for doc in tqdm(docs):
        # filename is copied from doc_id        
        doc_id = doc.attrib['id']
        # get texts in each paragraph
        paragraphs = doc.findall('.//P')
        labeled_sents = [str(p.text) for p in paragraphs if p.text != '\n']
        processed_doc = ' '.join([extract_words(sent) for sent in labeled_sents])
        # doc_dict[doc_id] = processed_doc
        prefix = doc_id.split('.')[0][:-2]
        nyt_extracted_subdir = os.path.join(nyt_extracted_dir, prefix)
        verify_dir(nyt_extracted_subdir)
        extracted_filepath = os.path.join(nyt_extracted_subdir, doc_id + '.txt')
        if processed_doc == '':
            continue
        doc_id_set.add(doc_id)
        valid_count += 1
        with open(extracted_filepath, 'w') as out_f:
            out_f.write(processed_doc)
    
    filelist_path = os.path.join(nyt_extracted_subdir, 'filelist.txt')
    with open(filelist_path, 'w') as f:
        for doc_id in doc_id_set:
            f.write(doc_id + '.txt\n')

    print('complete parsing the xml file: {}'.format(xml_name))
    print('{} documents have been extracted from the xml file...'.format(str(valid_count)))


def main():
    parser = ArgumentParser()
    # parser.add_argument('nyt_filename', help='the Gigaword NYT XML filename')
    parser.add_argument('--nyt_xml_dir', help='Path to the Gigaword NYT XML file')
    parser.add_argument('--nyt_extracted_dir', help='Path to the extracted Gigaword NYT TXT file')
    
    args = parser.parse_args()
    verify_dir(args.nyt_extracted_dir)
    for filename in os.listdir(args.nyt_xml_dir):
        if filename.split('.')[-1] != 'xml':
            continue
        extract_nyt(filename, args.nyt_xml_dir, args.nyt_extracted_dir)


if __name__ == '__main__':
    main()

