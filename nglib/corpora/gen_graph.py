import os
import sys
import dill
import json
import string
from collections import OrderedDict
from argparse import ArgumentParser
from stanza.server import CoreNLPClient
from itertools import combinations
from tqdm import tqdm


class Indexer(object):
    """Word to index bidirectional mapping."""

    def __init__(self, start_symbol="<s>", end_symbol="</s>"):
        """Initializing dictionaries and (hard coded) special symbols."""
        self.word_to_index = {}
        self.index_to_word = {}
        self.size = 0
        # Hard-code special symbols.
        self.get_index("PAD", add=True)
        self.get_index("UNK", add=True)
        self.get_index(start_symbol, add=True)
        self.get_index(end_symbol, add=True)

    # Print size info
    def __repr__(self):
        return "This indexer currently has %d words" % self.size

    def get_word(self, index):
        """Get word by index if its in range. Otherwise return `UNK`."""
        return self.index_to_word[index] if index < self.size and index >= 0 else "UNK"

    def get_index(self, word, add):
        """Get index by word. If `add` is on, also append word to the dictionaries."""
        if self.contains(word):
            return self.word_to_index[word]
        elif add:
            self.word_to_index[word] = self.size
            self.index_to_word[self.size] = word
            self.size += 1
            return self.word_to_index[word]
        return self.word_to_index["UNK"]

    def contains(self, word):
        """Return True/False to indicate whether a word is in the dictionaries."""
        return word in self.word_to_index

    def add_sentence(self, sentence, add):
        """Add all the words in a sentence (a string) to the dictionary."""
        indices = [self.get_index(word, add) for word in sentence.split()]
        return indices

    def add_document(self, document_path, add):
        """Add all the words in a document (a path to a text file) to the dictionary."""
        indices_list = []
        with open(document_path, "r") as document:
            for line in document:
                indices = self.add_sentence(line, add)
                indices_list.append(indices)
        return indices_list

    def to_words(self, indices):
        """Indices (ints) -> words (strings) conversion."""
        return [self.get_word(index) for index in indices]

    def to_sent(self, indices):
        """Indices (ints) -> sentence (1 string) conversion."""
        return " ".join(self.to_words(indices))

    def to_indices(self, words):
        """Words (strings) -> indices (ints) conversion."""
        return [self.get_index(word, add=False) for word in words]


class Node:
    def __init__(self, doc_id, sentence_id, pos_id, node_word, node_type):
        # <doc_id> describes the ID belonging to the single document source the node came from 
        self.doc_id = doc_id
        # <sentence_id> describes the ID belonging to the sentence the node came from in the single document
        self.sentence_id = str(sentence_id)
        self.node_type = node_type
        assert self.node_type in ['event', 'token']
        # <pos_id> describes the position ID of the node in the sentence
        self.pos_id = str(pos_id) if self.node_type == 'token' else None
        self.node_word = node_word if self.node_type == 'token' else ('Event' + self.sentence_id)
        self.edge_ids = set()
        self.distinct_id = '_'.join([self.doc_id, self.sentence_id, self.pos_id, self.node_word, self.node_type])

    def __repr__(self):
        return "[Node] | ID: %s" % (self.distinct_id)

    @staticmethod
    def entry_type():
        return "Node"


class Edge:
    def __init__(self, doc_id, sentence_id, edge_type, edge_label, source_id, target_id):
        self.edge_type = edge_type
        assert self.edge_type in ['neighbor', 'dependency', 'coreference']
        self.doc_id = doc_id
        self.sentence_id = str(sentence_id)
        self.edge_label = edge_label
        self.source_id = source_id
        self.target_id = target_id
        if self.edge_type == 'coreference':
            self.distinct_id = '_'.join([self.doc_id, self.edge_label, self.source_id, self.target_id])
        else:
            self.distinct_id = '_'.join([self.doc_id, self.edge_label, self.sentence_id, self.source_id, self.target_id])

    def __repr__(self):
        return "[Edge] | ID: %s" % (self.distinct_id)

    def edge_info(self):
        return "Type: %s \nLabel: %s \nSource: %s \nTarget: %s" % (self.edge_type, self.edge_label, self.source_id, self.target_id)

    @staticmethod
    def entry_type():
        return "Edge"                


class Graph:
    def __init__(self, doc_id):
        self.doc_id = doc_id
        # A dictionary of Node objects
        self.nodes = dict()
        # A dictionary of Edge objects
        self.edges = dict()
        self.edge_info = {'neighbor': 0, 'dependency': 0, 'coreference': 0}

    def __repr__(self):
        return "[GRAPH] | ID: %s | #Nodes: %d | #Edges: %d" % (self.doc_id, len(self.nodes), len(self.edges))

    def is_empty(self):
        return len(self.nodes) == 0 and len(self.edges) == 0

    def get_node_id(self, sentence_id, pos_id):
        return [node_id for node_id in self.nodes.keys() if "_{}_{}_".format(sentence_id, pos_id) in node_id][0]

    def graph_info(self):
        self.edge_info['neighbor'] = len([edge_id for edge_id in self.edges if 'neighbor' in edge_id])
        self.edge_info['coreference'] = len([edge_id for edge_id in self.edges if 'coreference' in edge_id])
        self.edge_info['dependency'] = len(self.edges) - self.edge_info['neighbor'] - self.edge_info['coreference'] 


def load_nodes(graph, ann):
    prev_node = None
    for sentence in ann.sentence:
        for n in sorted(sentence.basicDependencies.node, key=lambda n: (n.sentenceIndex, n.index)):
            node = Node(graph.doc_id, n.sentenceIndex, n.index - 1, sentence.token[n.index - 1].word, sentence.token[n.index - 1].pos)
            graph.nodes[node.distinct_id] = node
            if n.index > 1:
                edge = Edge(graph.doc_id, n.sentenceIndex, 'neighbor', 'neighbor', node.distinct_id, prev_node.distinct_id)
                graph.edges[edge.distinct_id] = edge
                node.edge_ids.add(edge.distinct_id)
                prev_node.edge_ids.add(edge.distinct_id)
            prev_node = node


def load_edges(graph, ann):
    for sent_index, sentence in enumerate(ann.sentence):
        for e in sentence.basicDependencies.edge:
            source_id = graph.get_node_id(sent_index, e.source - 1)
            target_id = graph.get_node_id(sent_index, e.target - 1)
            edge = Edge(graph.doc_id, sent_index, 'dependency', e.dep, source_id, target_id)
            graph.edges[edge.distinct_id] = edge
            graph.nodes[source_id].edge_ids.add(edge.distinct_id)
            graph.nodes[target_id].edge_ids.add(edge.distinct_id)            
        
    coref_chains = {coref_chain.chainID: set() for coref_chain in ann.corefChain}
    for mention in ann.mentionsForCoref:
        if mention.corefClusterID not in coref_chains:
            continue
        coref_chains[mention.corefClusterID].add((mention.sentNum, mention.endIndex - 1))
    
    for _, value in coref_chains.items():
        comb = combinations(value, 2)
        for c in comb:
            source_id = graph.get_node_id(c[0][0], c[0][1])
            target_id = graph.get_node_id(c[1][0], c[1][1])
            edge = Edge(graph.doc_id, None, 'coreference', 'coreference', source_id, target_id)
            graph.edges[edge.distinct_id] = edge
            graph.nodes[source_id].edge_ids.add(edge.distinct_id)
            graph.nodes[target_id].edge_ids.add(edge.distinct_id)


def read_graph(doc_id, ann):
    # Construct Graph object
    graph = Graph(doc_id)
    load_nodes(graph, ann)
    load_edges(graph, ann)
    return graph


def verify_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_sent_predicates(sent, dep_key='enhancedPlusPlusDependencies'):
    '''get predicates for a sentence using our predicate definition: verb
    parameters:
        sent: Stanford CoreNLP sentence in json
    '''
    deps = sent[dep_key]
    sent_predicates = OrderedDict() # make it deterministic
    for dep in deps:
        if not dep['dep'].startswith(
                ('nsubj', 'dobj', 'iobj', 'nsubjpass',
                 'nmod', 'obj', 'pobj', 'obl', 'prep')):
            continue

        # predicate definition: verbs (exclude be-verbs) with interested edges
        pred_tidx = dep['governor'] - 1
        if not sent['tokens'][pred_tidx]['pos'].startswith('V'): # only verb events
            continue
        if sent['tokens'][pred_tidx]['lemma'] == 'be': # no be verbs
            continue

        if pred_tidx not in sent_predicates:
            sent_predicates[pred_tidx] = {}

        if dep['dep'].startswith('nsubjpass'): # exclude this from nsubj
            d = 'o'
        elif dep['dep'].startswith(('nsubj', 'nmod:agent', 'obl:agent')):
            d = 's'
        elif dep['dep'].startswith(('dobj', 'obj')):
            d = 'o'
        else: # iobj, nmod, pobj, obl, prep
            d = 'prep'
        if d in sent_predicates[pred_tidx]:
            # it is possible that a predicate has multiple subjects or objects.
            # we only take one of them for simplicity
            continue
        sent_predicates[pred_tidx][d] = dep['dependent'] - 1

    for dep in deps:
        if (dep['dep'] == 'compound:prt'
                and dep['governor'] - 1 in sent_predicates):
            pred_tidx = dep['governor'] - 1
            if pred_tidx not in sent_predicates:
                continue
            sent_predicates[pred_tidx]['prt'] = dep['dependent'] - 1
    return sent_predicates


def main():
    parser = ArgumentParser()
    parser.add_argument('--xml_dir', default='./data/xml/', help='Path to the Gigaword XML file')
    parser.add_argument('--doc_dir', default='./data/doc/', help='Path to the Gigaword Doc file')
    parser.add_argument('--graph_dir', default='./data/graph/', help='Path to the Gigaword Graph file')
    args = parser.parse_args()

    filename = './data/parse/nyt_eng_200001/NYT_ENG_20000101.0001'
    with open(filename, 'r') as f:
        ann = json.loads(f.read())

    sent_predicates = [get_sent_predicates(sent) for sent in ann['sentences']]
    print(ann.keys())

    # verify_dir(args.graph_dir)


"""
    doc_list = os.listdir(args.doc_dir)
    num_nodes, num_edges = [], []
    edge_info = {'neighbor': 0, 'dependency': 0, 'coreference': 0}

    properties = {'tokenize.whitespace': True, 'tokenize.keepeol': True, 'ssplit.eolonly': True, 'ner.useSUTime': False}
    annotators = ['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'depparse', 'coref']

    with CoreNLPClient(
            annotators=['tokenize','ssplit', 'depparse','coref'],
            timeout=300000,
            memory='16G') as client:
        # Use doc_name as graph_id
        for graph_id in tqdm(doc_list):
            doc_path = os.path.join(args.doc_dir, graph_id)
            with open(doc_path, 'r') as doc_file:
                doc = doc_file.read()
            doc = ' '.join([word for word in doc.split(' ') if word not in string.punctuation and word not in "``\'\'nn"]).strip()
            ann = client.annotate(doc, annotators=annotators, properties=properties, output_format='json')
            graph = read_graph(graph_id, ann)
            graph.graph_info()
            edge_info = {key: edge_info.get(key, 0) + graph.edge_info.get(key, 0) for key in edge_info.keys()}
            num_nodes.append(len(graph.nodes))
            num_edges.append(len(graph.edges))
            dill.dump(graph, open(os.path.join(args.graph_dir, graph_id + '.p'), 'wb'))

    print(str(len(doc_list)) + ' documents have been processed...')
    print('Avg. #Nodes: {}'.format(sum(num_nodes) / len(doc_list)))
    print('Avg. #Edges: {}'.format(sum(num_edges) / len(doc_list)))
    print('Avg. #Neighbor Edges: {}'.format(edge_info['neighbor'] / len(doc_list)))
    print('Avg. #Dependency Edges: {}'.format(edge_info['dependency'] / len(doc_list)))
    print('Avg. #Coreference Edges: {}'.format(edge_info['coreference'] / len(doc_list)))
"""


if __name__ == '__main__':
    main()