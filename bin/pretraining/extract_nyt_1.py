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


def extract_nyt(xml_name, xml_dir, nyt_dir):
    xml_path = os.path.join(xml_dir, xml_name)
    with open(xml_path, 'r') as xml_file:
        xml_tree = etree.fromstring(xml_file.read())
    docs = xml_tree.xpath('//DOC')
    del xml_tree

    doc_dict = {}
    for doc in tqdm(docs):
        # filename is copied from doc_id        
        doc_id = doc.attrib['id']
        # get texts in each paragraph
        paragraphs = doc.findall('.//P')
        labeled_sents = [str(p.text) for p in paragraphs if p.text != '\n']
        processed_doc = [extract_words(sent) for sent in labeled_sents]
        doc_dict[doc_id] = processed_doc

    print(str(len(docs)) + ' documents have been extracted from the xml file...')        
    
    json_name = xml_name[:-3] + 'json'
    nyt_path = os.path.join(nyt_dir, json_name)
    with open(nyt_path, 'w') as nyt_file:
        nyt_file.write(json.dumps(doc_dict))


def main():
    parser = ArgumentParser()
    parser.add_argument('nyt_filename', help='the Gigaword NYT XML filename')
    parser.add_argument('--nyt_dir', default='../../data/nyt/xml/', help='Path to the Gigaword NYT XML file')
    parser.add_argument('--extracted_nyt_dir', default='../../data/nyt/extracted_nyt/', help='Path to the extracted Gigaword NYT JSON file')
    
    args = parser.parse_args()
    verify_dir(args.extracted_nyt_dir)
    extract_nyt(args.nyt_filename, args.nyt_dir, args.extracted_nyt_dir)


if __name__ == '__main__':
    main()

