import os
import numpy as np
from datetime import datetime
import sys
from func import *

def loadFiles(file_list, catalog_list, inv_list):
    for f in file_list:
        index_no = f.split('_')[1]
        catalog_list.append(loadDict(DIR_MERGE, f, [0,1,2]))
        inv_list.append(open(DIR_MERGE + INV_FILE + '_' + index_no + '.txt', 'rb'))
    return

def matchScore(term, matching_scores, query_tf):
    for cat, inv_file in zip(catalog_list, inv_list):
        if term in cat:
            offset, length = cat[term]
            inv_file.seek(offset)
            tf_line = inv_file.readline()
            # term_id, {'df': df, 'ttf': ttf, 'info': [[doc_id, tf, [pos]]]}
            term_id, tf_info = loadTFInfo(tf_line)
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
    # tf_info: {'df': df, 'ttf': ttf, 'info': [[doc_id, tf, [pos]]]}
    method = 'tfidf'
    avg_doc_length = np.mean(map(lambda x: x[1][1] , DOC_MAP.items()))
    total_doc = STATS['D']
    for doc_info in tf_info['info']:
        doc_id, w_tf = doc_info[0], doc_info[1]
        dl = DOC_MAP[doc_id][1]
        w_df = tf_info['df']
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
    # tf_info: {'df': df, 'ttf': ttf, 'info': [[doc_id, tf, [pos]]]}
    method = 'okapi_bm25'
    avg_doc_length = np.mean(map(lambda x: x[1][1] , DOC_MAP.items()))
    total_doc = STATS['D']
    for doc_info in tf_info['info']:
        doc_id, w_tf = doc_info[0], doc_info[1]
        dl = DOC_MAP[doc_id][1]
        w_df = tf_info['df']
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
    # tf_info: {'df': df, 'ttf': ttf, 'info': [[doc_id, tf, [pos]]]}
    method = 'unigramJM'
    vocabulary = STATS['V']
    for doc_info in tf_info['info']:
        doc_id, w_tf = doc_info[0], doc_info[1]
        dl = DOC_MAP[doc_id][1]
        w_cf = tf_info['ttf']
        p_jm = lamb * w_tf / dl + (1 - lamb) * w_cf / vocabulary
        score = np.log(p_jm)
        if doc_id in matching_scores[method]:
            matching_scores[method][doc_id][term] = score
        else:
            matching_scores[method][doc_id] = {term: score}
    return

def readQuery(line, queries, stemmed_method):
    query_no = line[0]
    queries[query_no] = {}
    for w in line[1:]:
        w = stemWord(w, stemmed_method, STOP_WORDS)
        if w == '' or w not in TERM_ID:
            continue
        w_id = TERM_ID[w]
        if w in queries[query_no]:
            queries[query_no][w_id] += 1
        else:
            queries[query_no][w_id] = 1
    return

def aggScore(score, method, query = None, jm_score = None): # [doc_id, {term:score}]
    score_sum = sum(map(lambda x: x[1], score[1].items()))
    if method == 'unigramJM':
        for term in query:
            if term not in score[1]:
                score_sum += jm_score[term]
    return [score[0], score_sum]

def getJMMinScore(query_terms):
    vocabulary = STATS['V']
    lamb = .9
    jm_score = {}
    for term in query_terms:
        for cat, inv_file in zip(catalog_list, inv_list):
            if term in cat:
                offset, length = cat[term]
                inv_file.seek(offset)
                tf_line = inv_file.readline()
                # term_id, {'df': df, 'ttf': ttf, 'info': [[doc_id, tf, [pos]]]}
                term_id, tf_info = loadTFInfo(tf_line)
        w_cf = tf_info['ttf']
        p_jm = (1 - lamb) * w_cf / vocabulary
        jm_score[term] = np.log(p_jm)
    return jm_score

args = sys.argv
if len(args) == 1:
    stemmed_method = 'no_stop_words'
else:
    stemmed_method = args[1]

DIR_DATA = '../data/' + stemmed_method + '/'
DIR_MERGE = DIR_DATA + 'merged_indexing_files/'

STATS = loadDict(DIR_DATA, 'STATS', [1])
TERM_MAP = loadDict(DIR_DATA, 'TERM_MAP', [0])
DOC_MAP = loadDict(DIR_DATA, 'DOC_MAP', [0,2])
TERM_ID = {}
for term_id, term in TERM_MAP.iteritems():
    TERM_ID[term] = term_id

QUERY_FILE = '../AP_DATA/query_desc.51-100.short_cut.txt'
STOP_WORDS = None
if stemmed_method in ['no_stop_words', 'stemmed_no_stop_words']:
    STOP_WORDS = loadStopWords()

print 'load queries'
cnt = 0
queries = {} # {query_no:{terms:tf}}
with open(QUERY_FILE) as f:
    for l in f:
        line = l.rstrip('\n').replace('. ', ' ').replace('\t', '')\
                    .replace('(', '').replace(')', '').split(' ')
        line = filter(lambda x: len(x) == 1 and x[0].isalnum() or
                                len(x) > 1 and x[1].isalnum(), line)
        if len(line) > 1 and line[0].isdigit():
            cnt += 1
            readQuery(line, queries, stemmed_method)
print 'total number of queries loaded is', cnt

CAT_FILE = 'CATALOG'
INV_FILE = 'INV'
file_list = filter(lambda f: f[:len(CAT_FILE)] == CAT_FILE,
                            os.listdir(DIR_MERGE))
file_list = sorted(map(lambda f: f.rstrip('.txt'), file_list),
                    key = lambda x: int(x.split('_')[1]))
print len(file_list), 'files to load'
catalog_list, inv_list = [], []
loadFiles(file_list, catalog_list, inv_list)

print 'match documents'
topK = 1000
result_files = {}
for matching_method in ['tfidf', 'okapi_bm25', 'unigramJM']:
    result_files[matching_method] = open(DIR_DATA + matching_method + '.txt', 'wb')

for query_no, query_tf in queries.iteritems(): # {query_no:{terms:tf}}
    query_terms = query_tf.keys()
    # matching_scores: {doc_id: {term:score}}
    matching_scores = {'tfidf':{}, 'okapi_bm25':{}, 'unigramJM':{}}
    now = datetime.now()
    print 'match documents for query', query_no
    print map(lambda x: TERM_MAP[x], query_terms)
    for term in query_terms:
        # print 'match term', term, TERM_MAP[term]
        matchScore(term, matching_scores, query_tf)
    print 'calculate score and rank'
    print len(matching_scores['tfidf'].keys()), 'documents match'

    for matching_method in matching_scores.keys():
        if matching_method == 'unigramJM':
            jm_score = getJMMinScore(query_terms)
            scores = map(lambda score: aggScore(score, matching_method,
                                                query_terms, jm_score),
                        matching_scores[matching_method].items())
        else:
            scores = map(lambda score: aggScore(score, matching_method),
                        matching_scores[matching_method].items())
        scores = sorted(scores, key = lambda x: -x[1])[:topK]
        rank = 1
        for score in scores:
            doc_no = DOC_MAP[score[0]][0]
            result_files[matching_method].write(' '.join([query_no, 'Q0', doc_no,
                                            str(rank), str(score[1]), 'Exp', '\n']))
            rank += 1
    print 'running time is ', datetime.now() - now

for matching_method in result_files.keys():
    result_files[matching_method].close()
