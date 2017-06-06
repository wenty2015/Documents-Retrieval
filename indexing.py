import os
import cPickle
from nltk.tokenize import TreebankWordTokenizer
from datetime import datetime
import sys
from stemming.porter2 import stem

def updateTerm(text, docno, term_dict, termid_dict, term_map, doc_map, stem_method,
                vocabulary, documents):
    terms = tokenizeText(text, stem_method) # {term: [position]}
    if len(terms.keys()) > 0: # non-empty text
        # update doc_map, {doc_id: [docno, doc length]}
        documents += 1
        doc_id = documents
        doc_length = sum(map(lambda x: len(x[1]), terms.items()))
        doc_map[documents] = [docno, doc_length]
        # update term_map
        for term, pos in terms.iteritems():
            if term not in termid_dict: # update term_map, {term_id: [term, df, ttf]}
                vocabulary += 1
                termid_dict[term] = vocabulary
                term_id = vocabulary
                term_map[term_id] = [term, 1, len(pos)] # [term, df, ttf]
            else:
                term_id = termid_dict[term]
                term_map[term_id][1] += 1 # df
                term_map[term_id][2] += len(pos) # ttf

            if term_id not in term_dict: # update term_dict
                term_dict[term_id] = [[doc_id, len(pos), pos]] # [docid, tf, [pos]]
            else:
                term_dict[term_id].append([doc_id, len(pos), pos])
    return vocabulary, documents

def stemWord(w, stem_method): # to do
    w = stemAsIs(w)
    if stem_method in ['no_stop_words', 'stemmed_no_stop_words']:
        if w in stop_words:
            w = ''
    if stem_method in ['stemmed', 'stemmed_no_stop_words']:
        w = stem(w)
    return w

def stemAsIs(w):
    return w.rstrip('=.-:\\').replace(',','').lower()

def tokenizeText(text, stem_method):
    tokens = TreebankWordTokenizer().tokenize(text)
    terms = {}
    position = 0
    for w in tokens:
        if not w[0].isalnum(): # remove the non-words
            continue
        w_stemmed = stemWord(w, stem_method)
        if w_stemmed == '':
            continue
        if w_stemmed not in terms:
            terms[w_stemmed] = [position]
        else:
            terms[w_stemmed].append(position)
        position += 1
    return terms

def dumpFile(term_dict, cnt): # term_dict: {term: [[docid, tf, [pos]]]}
    f_inv = open(DIR_DATA + 'indexing_files/INV_'+str(cnt)+'.txt', 'wb')
    catalog = {}
    for term, term_info in term_dict.iteritems():
        offset = f_inv.tell()
        sorted_info = sorted(term_info, key = lambda x: -x[1]) # ordered by tf desc
        for doc_info in sorted_info:
            text = '|' + ' '.join([str(doc_info[0]), str(doc_info[1]),
                                    ','.join(map(lambda x: str(x), doc_info[2]))])
            f_inv.write(text)
        f_inv.write('\n')
        length = f_inv.tell() - offset
        catalog[term] = {'t': [offset, length]}
    f_inv.close()
    with open(DIR_DATA + 'indexing_files/CATALOG_'+str(cnt), 'wb') as f_cat:
        cPickle.dump(catalog, f_cat)
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
    stem_method = 'as_is'
else:
    stem_method = args[1:]
DIR_DATA = '../data/' + stem_method + '/'

def loadStopWords():
    stop_words = set()
    with open('stoplist.txt', 'rb') as f:
        for line in f:
            stop_words.add(line.replace('\n', ''))
    return stop_words

if stem_method in ['no_stop_words', 'stemmed_no_stop_words']:
    stop_words = loadStopWords()

now = datetime.now()
for file_name in file_list:
    if file_name[:2].lower() != 'ap':
        continue
    with open(DIR + file_name) as f:
        for l in f:
            line = l.replace('\n','')
            if line[:len(TAG_DOC)] == TAG_DOC:
                if batch_cnt == BATCH:
                    dumpFile(term_dict, cnt) # dump file for a doc batch
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
    dumpFile(term_dict, cnt)
print 'total number of documents loaded is', cnt
print 'running time is ', datetime.now() - now

ttf = sum(map(lambda x: x[1][1], doc_map.items())) # {doc_id: [docno, doc length]}
stats = {'V': vocabulary, 'D': documents, 'TTF': ttf}
print 'writing STATS and MAP files'
now = datetime.now()
with open(DIR_DATA + 'term_dict', 'wb') as f:
    cPickle.dump(term_dict, f)
with open(DIR_DATA + 'TERM_ID', 'wb') as f:
    cPickle.dump(term_id, f)
with open(DIR_DATA + 'STATS', 'wb') as f:
    cPickle.dump(stats, f)
with open(DIR_DATA + 'TERM_MAP', 'wb') as f:
    cPickle.dump(term_map, f)
with open(DIR_DATA + 'DOC_MAP', 'wb') as f:
    cPickle.dump(doc_map, f)
print 'running time is ', datetime.now() - now
