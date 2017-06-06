import os
import cPickle
import numpy as np
from datetime import datetime

def loadFiles(file_list, catalog_list, inv_list):
    for f in file_list:
        index_no = f.split('_')[1]
        with open(DIR_MERGE + f, 'rb') as f:
            catalog_list.append(cPickle.load(f))
        inv_list.append(open(DIR_MERGE + INV_FILE + '_' + index_no + '.txt', 'rb'))
    return

def matchScore(term, matching_scores, query_tf):
    for cat, inv_file in zip(catalog_list, inv_list):
        if term in cat:
            offset, length = cat[term]['t']
            present_offset = offset
            while present_offset < offset + length:
                inv_file.seek(present_offset)
                tf_line = inv_file.readline()
                present_offset = inv_file.tell()
                tf_info = loadTFInfo(tf_line) # [doc_id, tf, [pos]]
                calculateScore(term, matching_scores, tf_info, query_tf)
    return

def calculateScore(term, matching_scores, tf_info, query_tf):
    tf_idf(term, matching_scores, tf_info)
    okapi_bm25(query_tf, term, matching_scores, tf_info)
    unigramJM(term, matching_scores, tf_info, lamb = .9)
    return

def tf_idf(term, matching_scores, tf_info):
    '''calculate similarity score for q and d, based on TF-IDF
    okapi_tf(w,d) = tf_{w,d}/(tf_{w,d} + 0.5 + 1.5 * len_d / avg(len_d))
    tfidf(d,q) = \sum_{w in q}{okapi_tf(w,d)*log(D/df_w)}
    '''
    method = 'tfidf'
    avg_doc_length = np.mean(map(lambda x: x[1][1] , DOC_MAP.items()))
    total_doc = STATS['D']
    for doc_info in tf_info:
        doc_id, w_tf = doc_info[0], doc_info[1]
        dl = DOC_MAP[doc_id][1]
        w_df = TERM_MAP[term][1]
        okapi_tf = w_tf/(w_tf + 0.5 + 1.5 * dl / avg_doc_length)
        score = okapi_tf * np.log(total_doc * 1.0 / w_df)
        if doc_id in matching_scores[method]:
            matching_scores[method][doc_id][term] = score
        else:
            matching_scores[method][doc_id] = {term: score}
    return

def okapi_bm25(query_tf, term, matching_scores, tf_info,
              k1 = 1.2, k2 = 100, b = 0.75):
    '''calculate similarity score for q and d, based on Okapi BM25
    okapi_bm25(w,d,q) = log((d + .5) / (df_w + .5)) *
                        (tf_{w,d} + k1 * tf_{w,d})/
                            (tf_{w,d} + k1 * (1 - b + b * len_d / avg(len_d))) *
                        (tf_{w,q} + k2 * tf_{w,q}) /
                            (tf_{w,q} + k2)
    bm25(d,q) = \sum_{w in q}{okapi_bm25(w,d,q)}
    '''
    method = 'okapi_bm25'
    avg_doc_length = np.mean(map(lambda x: x[1][1] , DOC_MAP.items()))
    total_doc = STATS['D']
    for doc_info in tf_info:
        doc_id, w_tf = doc_info[0], doc_info[1]
        dl = DOC_MAP[doc_id][1]
        w_df = TERM_MAP[term][1]
        q_tf = query_tf[term]

        idf = np.log((total_doc + .5) / (w_df + .5))
        tf_d = (w_tf + k1 * w_tf) /  (w_tf + k1 * (1 - b + b * dl / avg_doc_length))
        tf_q = (q_tf + k2 * q_tf) / (q_tf + k2)
        score = idf * tf_d * tf_q
        if doc_id in matching_scores[method]:
            matching_scores[method][doc_id][term] = score
        else:
            matching_scores[method][doc_id] = {term: score}
    return

def unigramJM(term, matching_scores, tf_info, lamb = .5):
    '''calculate similarity score for q and d, based on Unigram LM with
    Laplace smoothing
    p_laplace(w|d) = (tf_{w,d} + 1) / (len_d + V)
    lm_laplace(d,q) = \sum_{w in q} log(p_laplace(w|d))'''
    method = 'unigramJM'
    vocabulary = STATS['V']
    for doc_info in tf_info:
        doc_id, w_tf = doc_info[0], doc_info[1]
        dl = DOC_MAP[doc_id][1]
        w_cf = TERM_MAP[term][2]
        p_jm = lamb * w_tf / dl + (1 - lamb) * w_cf / vocabulary
        score = np.log(p_jm)
        if doc_id in matching_scores[method]:
            matching_scores[method][doc_id][term] = score
        else:
            matching_scores[method][doc_id] = {term: score}
    return

def loadTFInfo(tf_line):
    #input: '|doc_id tf pos,pos\n', output: [[doc_id, tf, [pos]]]
    docs = tf_line.rstrip('\n').split('|')[1:]
    docs = map(lambda x: loadDocInfo(x), docs)
    return docs

def loadDocInfo(doc_line): # 'doc_id tf pos,pos'
    doc = doc_line.split(' ')
    pos = map(lambda x: int(x), doc[2].split(','))
    doc = map(lambda x: int(x), doc[:2])
    doc.append(pos)
    return doc

with open('../data/STATS', 'rb') as f:
    STATS = cPickle.load(f)
with open('../data/TERM_MAP', 'rb') as f:
    TERM_MAP = cPickle.load(f)
with open('../data/DOC_MAP', 'rb') as f:
    DOC_MAP = cPickle.load(f)

DIR_MERGE = '../data/merged_indexing_files/'
CAT_FILE = 'CATALOG'
INV_FILE = 'INV'
file_list = sorted(filter(lambda f: f[:len(CAT_FILE)] == CAT_FILE, os.listdir(DIR_MERGE)),
                    key = lambda x: int(x.split('_')[1]))
print len(file_list), 'files to load'
catalog_list, inv_list = [], []
loadFiles(file_list, catalog_list, inv_list)

query = [12, 48]
query_tf = {12: 1, 48: 2}
matching_scores = {'tfidf':{}, 'okapi_bm25':{}, 'unigramJM':{}} # {doc_id: {term:score}}
now = datetime.now()
for term in query:
    matchScore(term, matching_scores, query_tf)
print 'running time is ', datetime.now() - now

with open('scores', 'wb') as f:
    cPickle.dump(matching_scores, f)
