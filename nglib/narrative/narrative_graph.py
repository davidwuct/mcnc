import os
import dill
import string
import logging
from tqdm import tqdm
from itertools import combinations
from collections import OrderedDict, defaultdict
from argparse import ArgumentParser

import json
import torch
import numpy as np
import dgl
from transformers import AutoTokenizer

import utils
import discourse as disc


MASK_DISC_CONN = '[MASK]'


class NGNode:
    def __init__(self, nid, ntag, ntype, name=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.nid = nid
        assert ntag in ['token', 'predicate']
        self.ntag = ntag
        self.ntype = ntype
        self.name = name
        self.conn = -1

        # (sent, pred_tidx, role_tidx)
        # self.instances = []
        # self.logger.debug('create node {}:{}:{}'.format(self.nid, self.ntype, self.rep_token))

    def __repr__(self):
        return '({}:{}:{})'.format(self.nid, self.name, self.ntype)
        # return '({}:{}:{}:{})'.format(
        #     self.nid, self.name, self.ntype, len(self.instances))


class NarrativeGraph:
    def __init__(self, doc_id, ntypes, rtypes, sentences, removed_conn=[]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.doc_id = doc_id
        self.nodes = {}
        self.nid2loc = {}
        self.edges = set()
        self.sentences = sentences
        self.removed_conn = removed_conn        
        if ntypes is None:
            self.logger.warning('ntype2idx is None. using default.')
            self.ntypes = {
                "CC": 0,
                "CD": 1,
                "DT": 2,
                "EX": 3,
                "FW": 4,
                "IN": 5,
                "JJ": 6,
                "JJR": 7,
                "JJS": 8,
                "LS": 9,
                "MD": 10,
                "NN": 11,
                "NNP": 12,
                "NNPS": 13,
                "NNS": 14,
                "PDT": 15,
                "POS": 16,
                "PRP": 17,
                "PRP$": 18,
                "RB": 19,
                "RBR": 20,
                "RBS": 21,
                "RP": 22,
                "SYM": 23,
                "TO": 24,
                "UH": 25,
                "VB": 26,
                "VBD": 27,
                "VBG": 28,
                "VBN": 29,
                "VBP": 30,
                "VBZ": 31,
                "WDT": 32,
                "WP": 33,
                "WP$": 34,
                "WRB": 35,
                "others": 36,
            }
        else:
            self.ntypes = ntypes
        if rtypes is None:
            self.logger.warning('rtype2idx is None. using default.')
            self.rtypes = {
                "next": 0,
                "cnext": 1,
                "Temporal.Asynchronous.Precedence": 2,
                "Temporal.Asynchronous.Succession": 3,
                "Temporal.Synchrony": 4,
                "Contingency.Cause.Reason": 5,
                "Contingency.Cause.Result": 6,
                "Comparison.Contrast": 7,
                "acl": 8,
                "advcl": 9,
                "advmod": 10,
                "amod": 11,
                "appos": 12,
                "aux": 13,
                "case": 14,
                "cc": 15,
                "ccomp": 16,
                "compound": 17,
                "conj": 18,
                "cop": 19,
                "csubj": 20,
                "dep": 21,
                "det": 22,
                "discourse": 23,
                "expl": 24,
                "fixed": 25,
                "goeswith": 26,
                "iobj": 27,
                "mark": 28,
                "nmod": 29,
                "nsubj": 30,
                "nummod": 31,
                "obj": 32,
                "obl": 33,
                "orphan": 34,
                "parataxis": 35,
                "punct": 36,
                "xcomp": 37,
                "others": 38,
            }
        else:
            self.rtypes = rtypes

    def add_node(self, n, sent_id, token_pos):
        if n.nid in self.nodes:
            self.logger.warning('failed to add node: {}'.format(n.nid))
            return False
        assert n.nid == len(self.nodes)
        self.nodes[n.nid] = n
        self.nid2loc[(sent_id, token_pos)] = n.nid

        return True

    def add_edge(self, nid1, nid2, rtype):
        if (nid1, nid2, rtype) in self.edges:
            self.logger.debug('failed add_edge: {} existed.'.format((nid1, nid2, rtype)))
            return False
        # self.logger.debug('add_edge: ({},{}): {}'.format(nid1, nid2, rtype))
        self.edges.add((nid1, nid2, rtype))
        return True

    def get_edge_stats(self):
        counts = {}
        for nid1, nid2, rtype in self.edges:
            if rtype not in counts:
                counts[rtype] = 0
            counts[rtype] += 1
        return counts

    def prune_graph(self):
        pnodes = {nid: n for nid, n in self.nodes.items() if n.ntag == 'predicate'}
        self.one_conn = {}
        for nid, n in pnodes.items():
            for (nid1, nid2, _) in self.edges:
                if nid == nid1 and nid2 not in self.one_conn:
                    self.one_conn[nid2] = self.nodes[nid2]
                elif nid == nid2 and nid1 not in self.one_conn:
                    self.one_conn[nid1] = self.nodes[nid1]
        self.one_conn = {nid: n for nid, n in self.one_conn.items() if n.ntag == 'token'}
        for nid, n in self.one_conn.items():
            n.conn = 1

        self.two_conn = {}
        for nid, n in self.one_conn.items():
            for (nid1, nid2, _) in self.edges:
                if nid == nid1 and nid2 not in self.one_conn and nid2 not in self.two_conn:
                    self.two_conn[nid2] = self.nodes[nid2]
                elif nid == nid2 and nid1 not in self.one_conn and nid1 not in self.two_conn:
                    self.two_conn[nid1] = self.nodes[nid1]
        self.two_conn = {nid: n for nid, n in self.two_conn.items() if n.ntag == 'token'}
        for nid, n in self.two_conn.items():
            n.conn = 2

        removed_nids = [nid for nid, n in self.nodes.items() if (n.ntag == 'token' and n.conn in self.removed_conn) 
                            or n.name in string.punctuation]
        [self.nodes.pop(nid) for nid in removed_nids]
        [self.nid2loc.pop(nid) for nid in removed_nids]
        
        removed_edges = set()
        for edge in self.edges:
            (nid1, nid2, _) = edge
            for nid in removed_nids:
                if nid1 == nid or nid2 == nid:
                    removed_edges.add(edge)
        self.edges -= removed_edges

    def get_bert_inputs(self, bert_tokenizer, max_seq_len):
        bert_inputs = bert_tokenizer(self.sentences, padding=True, return_tensors='pt')
        if bert_inputs['input_ids'].shape[1] > max_seq_len:
            return None, None, None
        input_ids = bert_inputs['input_ids']
        token_type_ids = bert_inputs['token_type_ids']
        attention_mask = bert_inputs['attention_mask']
        return input_ids, token_type_ids, attention_mask

    def to_dgl_inputs(self, prune_graph, bert_tokenizer, max_seq_len=64):
        if prune_graph: 
            self.prune_graph()
        # {"CC": {2: 0, 6: 1, ...}, "CD": {3: 0, 10: 1, ...}, ...}
        nid2tnid = defaultdict(dict)
        for nid, n in self.nodes:
            tnid = len(nid2tnid[n.ntype])
            nid2tnid[n.ntype][nid] = tnid

        """
        data_dict = {}
        for (nid1, nid2, rtype) in self.edges:
            ntype1 = self.nodes[nid1].ntype
            ntype2 = self.nodes[nid2].ntype
            tnid1 = nid2tnid[ntype1][nid1]
            tnid2 = nid2tnid[ntype2][nid2]
            if (ntype1, rtype, ntype2) not in data_dict:
                data_dict[(ntype1, rtype, ntype2)] = [[], []]
            data_dict[(ntype1, rtype, ntype2)][0].append(tnid1)
            data_dict[(ntype1, rtype, ntype2)][1].append(tnid2)            
        for canonical_edge in data_dict:
            srcidxs = torch.tensor(data_dict[canonical_edge][0])
            desidxs = torch.tensor(data_dict[canonical_edge][1])
            data_dict[canonical_edge] = (srcidxs, desidxs)
        g = dgl.heterograph(data_dict)
        """

        input_ids, token_type_ids, attention_mask = \
            self.get_bert_inputs(bert_tokenizer, max_seq_len)

        if input_ids is None:
            return None, None

        node_tags = [1 if self.nodes[nid].ntag == 'predicate' else 0 for nid in range(len(self.nodes))]
        node_types = [self.ntypes[n.ntype] for n in self.nodes.values()]
        edge_list = list(self.edges)
        nid1s = [e[0] for e in edge_list]
        nid2s = [e[1] for e in edge_list]
        edge_types = [self.rtypes[e[2]] for e in edge_list]
        edge_types = torch.from_numpy(np.array(edge_types))
        in_degrees = np.zeros((len(self.nodes), len(self.rtypes)))
        for (nid1, nid2, rtype) in self.edges:
            in_degrees[nid2][self.rtypes[rtype]] += 1
        edge_norms = np.zeros_like(in_degrees)
        np.reciprocal(in_degrees, where=in_degrees > 0.0, out=edge_norms)
        edge_norms = torch.from_numpy(edge_norms)

        bert_inputs = {
            'input_ids': torch.LongTensor(input_ids),
            'token_type_ids': torch.LongTensor(token_type_ids),
            'attention_mask': torch.LongTensor(attention_mask),
            'nid2loc': self.nid2loc,
        }
        rgcn_inputs = {
            'node_tags': node_tags,
            'node_types': node_types,
            'nid2tnid': nid2tnid,
            'edge_src': nid1s,
            'edge_dest': nid2s,
            'edge_types': edge_types,
            'edge_norms': edge_norms,
        }
        return bert_inputs, rgcn_inputs
        

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


def _events_in_arg(all_sent_predicates, arg):
    start_sent_idx, start_tok_idx, end_sent_idx, end_tok_idx = arg
    if start_sent_idx == end_sent_idx:
        events = [(start_sent_idx, pidx) for pidx in all_sent_predicates[start_sent_idx].keys()
                  if pidx >= start_tok_idx and pidx < end_tok_idx]
    else: # this is only possible for arg1
        # all events in the prev sentence
        prev_events = [(start_sent_idx, pidx) for pidx in all_sent_predicates[start_sent_idx].keys()]
        # cur sentense before marker
        events = [(end_sent_idx, pidx) for pidx in all_sent_predicates[end_sent_idx].keys()
                  if pidx >= 0 and pidx < end_tok_idx]
        events = events + prev_events
    return events


def _disc_relations(all_sent_predicates, arg1, arg2, rtype, parse):
    rels = []
    e1s = _events_in_arg(all_sent_predicates, arg1)
    e2s = _events_in_arg(all_sent_predicates, arg2)
    if len(e1s) > 0 and len(e2s) > 0:
        # we get all possible pairs
        for e1 in e1s:
            for e2 in e2s:
                # (rtype, i_sent1, pred_tidx1, i_sent2, pred_tidx2)
                r = (rtype, e1[0], e1[1], e2[0], e2[1])
                rels.append(r)
    return rels


def get_disc_relations(all_sent_predicates, parse, dmarkers):
    disc_rels = []
    for i_sent, sent in enumerate(parse['sentences']):
        matched = disc._find_matched_markers(sent, dmarkers)
        if matched is None:
            continue
        # (i_sent, begin_tidx, end_tidx, rtype, dmarker)
        conn = (i_sent, matched[0], matched[1], matched[2], matched[3])
        arg1, arg2 = disc._find_args(parse, conn)
        if arg1 is None:
            continue

        # event pairs
        # [(rtype, i_sent1, pred_tidx1, i_sent2, pred_tidx2), ...]
        rels = _disc_relations(all_sent_predicates, arg1, arg2, conn[3], parse)
        if len(rels) > 0:
            disc_rels += rels
    return disc_rels


def get_next_relations(all_sent_predicates):
    rels = []
    for i in range(len(all_sent_predicates)-1):
        cur_predicates = all_sent_predicates[i]
        next_predicates = all_sent_predicates[i+1]

        for pred_tidx1, deps1 in cur_predicates.items():
            for pred_tidx2, deps2 in next_predicates.items():
                rels.append((i, pred_tidx1, i+1, pred_tidx2))
    return rels


def predicate_rep(parse, i_sent, pred_tidx, deps):
    tok = parse['sentences'][i_sent]['tokens'][pred_tidx]
    if 'prt' in deps:
        prt_tok = parse['sentences'][i_sent]['tokens'][deps['prt']]
        rep = '{}_{}'.format(tok['lemma'], prt_tok['lemma'])
    else:
        rep = tok['lemma']
    return rep


def entity_rep(parse, i_sent, pred_tidx, deps, entity_dep):
    pred_rep = predicate_rep(parse, i_sent, pred_tidx, deps)
    return '{}-{}'.format(pred_rep, entity_dep)


def find_chain(i_sent, tidx, corefs):
    for cid, chain in corefs.items():
        for m in chain:
            cur_sent_idx = m['sentNum'] - 1
            cur_tok_start_idx = m['startIndex'] - 1
            cur_tok_end_idx = m['endIndex'] - 1
            if (i_sent == cur_sent_idx
                    and tidx >= cur_tok_start_idx
                    and tidx < cur_tok_end_idx):
                return cid
    return None


def get_ng_bert_tokenizer(bert_weight_name, is_cased):
    bert_tokenizer = AutoTokenizer.from_pretrained(
        bert_weight_name, do_lower_case=(not is_cased))
    return bert_tokenizer


def replace_discourse_markers(sent_parse, dmarkers):
    match = disc._find_matched_markers(
        sent_parse, dmarkers)
    toks = [t['word'] for t in sent_parse['tokens']]
    if match is None:
        return ' '.join(toks)

    begin_tidx = match[0]
    end_tidx = match[1]
    for i in range(begin_tidx, end_tidx):
        toks[i] = MASK_DISC_CONN
    return ' '.join(toks)


def get_pp_ridx2distr_coref(config):
    rtype2idx = config['rtype2idx']
    ret = {}
    for rtype, idx in rtype2idx.items():
        if rtype == 'cnext':
            ret[idx] = 1.0
        else:
            ret[idx] = 0.0
    return ret


def get_pp_ridx2distr(config):
    rtype2idx = config['rtype2idx']
    distr = config['pp_sampling_distribution']
    ret = {}
    for rtype, p in distr.items():
        ret[rtype2idx[rtype]] = p
    return ret


def create_narrative_graph(parse,
                           sent_predicates=None,
                           next_rels=None,
                           ntypes=None,
                           rtypes=None,
                           dmarkers=None,
                           disc_rels=None,
                           ):
    '''
    parameters:
        parse: CoreNLP
        sent_predicates: list of sentence dict
                        {pred_tidx: {'s': s_tidx, 'o': o_tidx, 'prt': prt_tidx,
                                    'name': 'e1', 'prep': prep_tidx}}
        next_rels: list of (sent_idx1, pred_tidx1, sent_idx2, pred_tidx2)
        rtypes: {"before": 0, ...}
        dmarkers: {"before": "Temporal.Asynchronous.Precedence", ...}
    '''
    doc_id = parse['doc_id']
    corefs = parse['corefs']
    g = NarrativeGraph(doc_id, ntypes=ntypes, rtypes=rtypes)

    if sent_predicates is None:
        sent_predicates = [get_sent_predicates(sent) for sent in parse['sentences']]
    if next_rels is None:
        next_rels = get_next_relations(sent_predicates)
    if dmarkers is not None and disc_rels is None:
        # event pairs: [(rtype, i_sent1, pred_tidx1, i_sent2, pred_tidx2), ...]
        disc_rels = get_disc_relations(sent_predicates, parse, dmarkers)

    # doc token nodes
    cur_nid = 0
    loc2node = {}
    for i_sent, sent in enumerate(parse['sentences']):
        for token in sent['tokens']:
            if token['pos'] in g.ntypes:
                n = NGNode(cur_nid, 'token', token['pos'], token['word'])
            else:
                n = NGNode(cur_nid, 'token', 'others', token['word'])
            loc2node[(i_sent, token['index']-1)] = n
            g.add_node(n, i_sent, token['index']-1)
            cur_nid += 1
            
    # doc predicate nodes
    for i_sent, predicates in enumerate(sent_predicates):
        for pred_tidx, _ in predicates.items():
            n = loc2node[(i_sent, pred_tidx)]
            n.ntag = 'predicate'

    # dependency parsing relations
    for i_sent, sent in enumerate(parse['sentences']):
        for dep in sent['basicDependencies']:
            if dep['dep'] == 'ROOT':
                continue
            n1 = loc2node[(i_sent, dep['governor']-1)]
            n2 = loc2node[(i_sent, dep['dependent']-1)]
            if dep['dep'].split(':')[0] in g.rtypes:
                g.add_edge(n1.nid, n2.nid, dep['dep'].split(':')[0])
            else:
                g.add_edge(n1.nid, n2.nid, 'others')

    # next relations
    for i_sent1, pred_tidx1, i_sent2, pred_tidx2 in next_rels:
        n1 = loc2node[(i_sent1, pred_tidx1)]
        n2 = loc2node[(i_sent2, pred_tidx2)]
        assert n1.ntag == 'predicate' and n2.ntag == 'predicate'
        g.add_edge(n1.nid, n2.nid, 'next')

    # discourse relations
    if disc_rels is not None:
        for rtype, i_sent1, pred_tidx1, i_sent2, pred_tidx2 in disc_rels:
            n1 = loc2node[(i_sent1, pred_tidx1)]
            n2 = loc2node[(i_sent2, pred_tidx2)]
            assert n1.ntag == 'predicate' and n2.ntag == 'predicate'
            g.add_edge(n1.nid, n2.nid, rtype)

    """
    # coref_next relations
    for _, coref in corefs.items():
        comb = combinations([idx for idx in range(len(coref))], 2)
        for c in list(comb):
            i_sent1, pred_tidx1 = coref[c[0]]['sentNum'] - 1, coref[c[0]]['endIndex'] - 2
            i_sent2, pred_tidx2 = coref[c[1]]['sentNum'] - 1, coref[c[1]]['endIndex'] - 2
            n1 = loc2node[(i_sent1, pred_tidx1)]
            n2 = loc2node[(i_sent2, pred_tidx2)]
            g.add_edge(n1.nid, n2.nid, 'coref')
    """

    # coref_next relations
    chainid2plocs = {}
    for i_sent, predicates in enumerate(sent_predicates):
        for pred_tidx, deps in predicates.items():
            for d, ent_tidx in deps.items():
                if d == 'prt' or d == 'name':
                    continue
                chain_id = find_chain(i_sent, ent_tidx, corefs)
                # for cnext
                if chain_id is not None:
                    if chain_id not in chainid2plocs:
                        chainid2plocs[chain_id] = set()
                    chainid2plocs[chain_id].add((i_sent, pred_tidx, d))

    for chain_id, plocs in chainid2plocs.items():
        coref_ploc_list = list(plocs)
        coref_ploc_list = sorted(coref_ploc_list, key=lambda x: (x[0], x[1]))
        if len(coref_ploc_list) <= 1:
            continue

        # design choice: coref_next window=1
        for i in range(len(coref_ploc_list)-1):
            n1 = loc2node[coref_ploc_list[i][:2]]
            n2 = loc2node[coref_ploc_list[i+1][:2]]
            if n1.nid == n2.nid:
                # two mentions that connects to the same predicate
                # refer to the same entity
                # print('same predicate coref mentions. drop it.')
                continue
            assert n1.ntag == 'predicate' and n2.ntag == 'predicate'
            g.add_edge(n1.nid, n2.nid, 'cnext')
    return g


def prepare_ng_model_inputs(bert_inputs, rgcn_inputs):
    edge_src = torch.LongTensor(rgcn_inputs['edge_src'])
    edge_types = torch.LongTensor(rgcn_inputs['edge_types'])
    edge_dest = torch.LongTensor(rgcn_inputs['edge_dest'])
    # back_edge_norms = torch.FloatTensor(rgcn_inputs['edge_norms']).unsqueeze(1)

    input_ids = bert_inputs['input_ids']
    input_masks = bert_inputs['input_masks']
    token_type_ids = bert_inputs['token_type_ids']
    target_idxs = bert_inputs['target_idxs']

    nid2rows = torch.from_numpy(pad_nid2rows(bert_inputs['nid2rows']))

    n_nodes = nid2rows.shape[0]
    n_instances = [input_ids.shape[0]]

    g = dgl.DGLGraph()
    g.add_nodes(n_nodes)
    g.add_edges(edge_src, edge_dest)
    edge_norms = []
    for i in range(edge_dest.shape[0]) :
        nid2 = int(edge_dest[i])
        edge_norms.append(1.0 / g.in_degree(nid2))
    edge_norms = torch.FloatTensor(edge_norms).unsqueeze(1)
    g.edata.update({'rel_type': edge_types})
    g.edata.update({'norm': edge_norms})

    inputs = {
        'bg': [[g]],
        'input_ids': input_ids,
        'input_masks': input_masks,
        'token_type_ids': token_type_ids,
        'target_idxs': target_idxs,
        'nid2rows': [nid2rows],
        'n_instances': n_instances
    }
    return inputs


def get_marker2rtype(rtype2markers):
    dmarkers = {}
    for rtype, markers in rtype2markers.items():
        for m in markers:
            dmarkers[m] = rtype
    return dmarkers


def main():
    parser = ArgumentParser()
    parser.add_argument('--xml_dir', default='./data/xml/', help='Path to the Gigaword XML file')
    parser.add_argument('--doc_dir', default='./data/doc/', help='Path to the Gigaword Doc file')
    parser.add_argument('--graph_dir', default='./data/graph/', help='Path to the Gigaword Graph file')
    args = parser.parse_args()

    filename = './data/parse/nyt_eng_200001/NYT_ENG_20000101.0001'
    with open(filename, 'r') as f:
        parse = json.loads(f.read())

    config = json.load(open('./data/config_narrative_graph.json', 'r'))
    dmarkers = get_marker2rtype(config["discourse_markers"])


    properties = {'tokenize.whitespace': True, 'tokenize.keepeol': True, 'ssplit.eolonly': True, 'ner.useSUTime': False}
    annotators = ['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'depparse', 'coref']    
    with CoreNLPClient(
            annotators=annotators,
            timeout=300000,
            memory='16G') as client:
        doc = "Jenny woke up very early and had some time to kill .\n She went outside and noticed that it was raining , so she went inside her favorite coffee-shop .\n She greeted the waiter ."
        # doc = "Okun was the production manager on the film 's Montana prairie set , while Benjamin , as United Artists ' general counsel , was in New York , sifting through the mounting bills and progress reports and eventually declaring director Michael Cimino in breach of contract .\n One week before `` Heaven 's Gate '' was set to begin filming in the remote Montana town of Kalispell , Cimino won the 1979 Academy Award as best director of `` The Deer Hunter .\n '' And Cimino , even after `` The Deer Hunter , '' did not feel that he was accepted by the film industry , so he was going to make the greatest film there is .\n Told he could not cast French actress Isabelle Huppert in the role of a brothel madam , he did so anyway .\n He installed his girlfriend Joann Carelli as the movie 's producer , assuring that he maintained total control of the production .\n He repeatedly shot and re-shot scenes , sometimes as many as 50 takes ."
        parse = client.annotate(doc, annotators=annotators, properties=properties, output_format='json')
        parse['doc_id'] = 0


    g = create_narrative_graph(parse, dmarkers=dmarkers)
    g.prune_graph()
    print(g.get_edge_stats())
    print(len(g.nodes), len(g.pnodes), len(g.one_conn), len(g.two_conn), len(g.edges))
    dill.dump(g, open('ng_example2_pruned.p', 'wb'))

    """
    pos_dict = {}
    basic_rel_dict, enhanced_rel_dict = {}, {}

    parse_dir = 'parse/nyt_eng_200001/'
    parse_list = os.listdir(parse_dir)
    for parse_filename in tqdm(parse_list):
        parse_path = os.path.join(parse_dir, parse_filename)
        with open(parse_path, 'r') as f:
            parse = json.loads(f.read()) 

        for parse_sentence in parse['sentences']:
            for token in parse_sentence['tokens']:
                if token['pos'] not in pos_dict:
                    pos_dict[token['pos']] = 0
                pos_dict[token['pos']] += 1

        for parse_sentence in parse['sentences']:
            for dep in parse_sentence['basicDependencies']:
                if dep['dep'].split(":")[0] not in basic_rel_dict:
                    basic_rel_dict[dep['dep'].split(":")[0]] = 0
                basic_rel_dict[dep['dep'].split(":")[0]] += 1
            for dep in parse_sentence['enhancedPlusPlusDependencies']:
                if dep['dep'].split(":")[0] not in enhanced_rel_dict:
                    enhanced_rel_dict[dep['dep'].split(":")[0]] = 0
                enhanced_rel_dict[dep['dep'].split(":")[0]] += 1

    pos_dict['all'] = sum(pos_dict.values())
    basic_rel_dict['all'] = sum(basic_rel_dict.values())
    enhanced_rel_dict['all'] = sum(enhanced_rel_dict.values())

    pos_dict = OrderedDict(sorted(pos_dict.items()))
    basic_rel_dict = OrderedDict(sorted(basic_rel_dict.items()))    
    enhanced_rel_dict = OrderedDict(sorted(enhanced_rel_dict.items()))

    print(pos_dict)
    print('\n')
    print(basic_rel_dict)
    print('\n')

    pos_dict = {k: v / pos_dict['all'] for k, v in pos_dict.items() if k != 'all'}
    basic_rel_dict = {k: v / basic_rel_dict['all'] for k, v in basic_rel_dict.items() if k != 'all'}
    enhanced_rel_dict = {k: v / enhanced_rel_dict['all'] for k, v in enhanced_rel_dict.items() if k != 'all'}

    pos_dict = OrderedDict(sorted(pos_dict.items()))
    basic_rel_dict = OrderedDict(sorted(basic_rel_dict.items()))    
    enhanced_rel_dict = OrderedDict(sorted(enhanced_rel_dict.items()))

    print(pos_dict)
    print('\n')
    print(basic_rel_dict)
    print('\n')
    # print(enhanced_rel_dict)
    print(len(pos_dict), len(basic_rel_dict), len(enhanced_rel_dict))
    """


if __name__ == '__main__':
    main()