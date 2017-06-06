import cPickle
import numpy as np

def aggScore(score, method, query = None): # [doc_id, {term:score}]
    score_sum = sum(map(lambda x: x[1], score[1].items()))
    if method == 'unigramJM':
        vocabulary = STATS['V']
        lamb = .9
        for term in query:
            if term not in score[1]:
                w_cf = TERM_MAP[term][2]
                p_jm = (1 - lamb) * w_cf / vocabulary
                score_sum += np.log(p_jm)
    return [score[0], score_sum]

with open('../data/STATS', 'rb') as f:
    STATS = cPickle.load(f)
with open('../data/TERM_MAP', 'rb') as f:
    TERM_MAP = cPickle.load(f)
with open('../data/DOC_MAP', 'rb') as f:
    DOC_MAP = cPickle.load(f)

with open('scores', 'rb') as f:
    matching_scores = cPickle.load(f)

topK = 1000
query = [12, 48]
query_length = len(query)
query_docno = '1'
for method in matching_scores.keys():
    f = open('../results/' + method + '.txt', 'wb')
    if method == 'unigramJM':
        scores = map(lambda score: aggScore(score, method, query), matching_scores[method].items())
    else:
        scores = map(lambda score: aggScore(score, method), matching_scores[method].items())
    scores = sorted(scores, key = lambda x: -x[1])[:topK]
    rank = 1
    for score in scores:
        doc_no = DOC_MAP[score[0]][0]
        f.write(' '.join([query_docno, 'Q0', doc_no,
                            str(rank), str(score[1]), 'Exp', '\n']))
        rank += 1
    f.close()
