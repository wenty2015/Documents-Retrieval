import os
from datetime import datetime
import sys
from func import *
from numpy import mean

def updateTerm(text, docno, term_dict, termid_dict, term_map, doc_map, stem_method,
                vocabulary, documents):
    terms = tokenizeText(text, stem_method) # {term: [position]}
    if len(terms.keys()) > 0: # non-empty text
        # update doc_map, {doc_id: [docno, doc length]}
        documents += 1
        doc_id = documents
        doc_length = sum(map(lambda x: len(x[1]), terms.items()))
        doc_map[documents] = [docno, doc_length]
        # update term_map and term_dict
        for term, pos in terms.iteritems():
            if term not in termid_dict: # update term_map, {term_id: term}
                vocabulary += 1
                termid_dict[term] = vocabulary
                term_id = vocabulary
                term_map[term_id] = term
            else:
                term_id = termid_dict[term]

            if term_id not in term_dict: # update term_dict
                term_dict[term_id] = {'df': 1, 'ttf': len(pos),
                                    'info': [[doc_id, len(pos), pos]]}
            else:
                term_dict[term_id]['df'] += 1
                term_dict[term_id]['ttf'] += len(pos)
                term_dict[term_id]['info'].append([doc_id, len(pos), pos])
    return vocabulary, documents

def tokenizeText(text, stem_method):
    tokens = tokenizer(text)
    terms = {}
    position = 0
    for w in tokens:
        if not w[0].isalnum(): # remove the non-words
            continue
        w_stemmed = stemWord(w, stem_method, STOP_WORDS)
        if w_stemmed == '':
            continue
        if w_stemmed not in terms:
            terms[w_stemmed] = [position]
        else:
            terms[w_stemmed].append(position)
        position += 1
    return terms

def dumpFile(term_dict, cnt, dir_file):
    # term_dict: {term: {'df': df, 'ttf': ttf, 'info':[[docid, tf, [pos]]]}}
    f_inv = open(dir_file + 'INV_'+str(cnt)+'.txt', 'wb')
    catalog = {}
    for term, term_info in term_dict.iteritems():
        offset, length = dumpTerm(term, term_info, f_inv)
        catalog[term] = [offset, length]
    f_inv.close()
    dumpDict(dir_file, 'CATALOG_'+ str(cnt), catalog)
    return

DIR = '../AP_DATA/ap89_collection/'
file_list = os.listdir(DIR)
print len(file_list), ' files to load'

TAG_DOC = '<DOC>'
TAG_DOCNO_START, TAG_DOCNO_END = '<DOCNO>', '</DOCNO>'
TAG_TEXT_START, TAG_TEXT_END = '<TEXT>', '</TEXT>'
BATCH = 1000
cnt, batch_cnt = 0, 0
term_dict, term_map, doc_map = {}, {}, {}
term_id = {}
vocabulary, documents, ttf = 0, 0, 0

args = sys.argv
if len(args) == 1:
    stem_method = 'no_stop_words'
else:
    stem_method = args[1]
DIR_DATA = '../data/' + stem_method + '/'
DIR_FILE = 'indexing_files/'
STOP_WORDS = None
if stem_method in ['no_stop_words', 'stemmed_no_stop_words']:
    STOP_WORDS = loadStopWords()

now = datetime.now()
for file_name in file_list:
    if file_name[:2].lower() != 'ap':
        continue
    with open(DIR + file_name) as f:
        for l in f:
            line = l.replace('\n','')
            if line[:len(TAG_DOC)] == TAG_DOC:
                if batch_cnt == BATCH:
                    dumpFile(term_dict, cnt, DIR_DATA + DIR_FILE) # dump file for a doc batch
                    batch_cnt = 0
                    term_dict = {}
                elif batch_cnt > 0:
                    vocabulary, documents = updateTerm(text, docno,
                        term_dict, term_id,
                        term_map, doc_map, stem_method, vocabulary, documents)
                read_text = False
                text = ''

                cnt, batch_cnt = cnt + 1, batch_cnt + 1
                if cnt % 5000 == 0:
                    print 'loaded ', cnt, ' documents'
            elif line[:len(TAG_DOCNO_START)] == TAG_DOCNO_START:
                docno = line.lstrip(TAG_DOCNO_START).rstrip(TAG_DOCNO_END).strip(' ')
            elif line[:len(TAG_TEXT_START)] == TAG_TEXT_START: # start of text
                read_text = True
                if len(line.lstrip(TAG_TEXT_START)) > 0:
                    text += l.lstrip(TAG_TEXT_START)
            elif line[-len(TAG_TEXT_END):] == TAG_TEXT_END: # end of text
                read_text = False
            else:
                if read_text:
                    text += l
if batch_cnt > 0:
    vocabulary, documents = updateTerm(text, docno, term_dict, term_id,
                    term_map, doc_map, stem_method, vocabulary, documents)
    dumpFile(term_dict, cnt, DIR_DATA + DIR_FILE)
print 'total number of documents loaded is', cnt
print 'running time is ', datetime.now() - now

ttf = sum(map(lambda x: x[1][1], doc_map.items())) # {doc_id: [docno, doc length]}
print 'V', vocabulary, 'D', documents, 'TTF', ttf, \
        'DL', mean(map(lambda x: x[1][1], doc_map.items()))
stats = {'V': vocabulary, 'D': documents, 'TTF': ttf}
print 'writing STATS and MAP files'
now = datetime.now()

dumpDict(DIR_DATA, 'STATS', stats, list_flag = False)
dumpDict(DIR_DATA, 'TERM_MAP', term_map, list_flag = False)
dumpDict(DIR_DATA, 'DOC_MAP', doc_map)
print 'running time is ', datetime.now() - now
