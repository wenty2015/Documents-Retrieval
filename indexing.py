import os
import json
from nltk.tokenize import TreebankWordTokenizer
from datetime import datetime

dir_data = '../AP_DATA/ap89_collection/'
file_list = os.listdir(dir_data)
print len(file_list), ' files to load'

def updateTerm(text, docno, term_dict, termid_dict, term_map, doc_map, stem_method,
                vocabulary, documents, ttf):
    terms = tokenizeText(text, stem_method) # {term: [position]}
    if len(terms.keys()) > 0: # non-empty text
        # update doc_map
        documents += 1
        doc_id = documents
        doc_length = sum(map(lambda x: len(x[1]), terms.items()))
        doc_map[documents] = [docno, doc_length]
        # update term_map
        for term, pos in terms.iteritems():
            if term not in termid_dict: # update term_map
                vocabulary += 1
                termid_dict[term] = vocabulary
                term_id = vocabulary
                term_map[term_id] = [term, 1, len(pos)] # [term, df, ttf]
            else:
                term_id = termid_dict[term]
                term_map[term_id][1] += 1 # df
                term_map[term_id][2] += len(pos) # ttf

            if term not in term_dict: # update term_dict
                term_dict[term_id] = [[doc_id, len(pos), pos]] # [docid, tf, [pos]]
            else:
                term_dict[term_id].append([doc_id, len(pos), pos])
            ttf += len(pos)
    return vocabulary, documents, ttf

def stemWord(w, stem_method): # to do
    if stem_method == 'as_is':
        return stemAsIs(w)
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
        if w_stemmed not in terms:
            terms[w_stemmed] = [position]
        else:
            terms[w_stemmed].append(position)
        position += 1
    return terms

def dumpFile(term_dict, cnt): # term_dict: {term: [[docid, tf, [pos]]]}
    f_inv = open('../data/INV_'+str(cnt)+'.txt', 'wb')
    catalog = {}
    for term, term_info in term_dict.iteritems():
        offset = f_inv.tell()
        sorted_info = sorted(term_info, lambda x: -x[1]) # ordered by tf desc
        for doc_info in sorted_info:
            text = '|' + ' '.join([str(doc_info[0]), str(doc_info[1]),
                                    ','.join(map(lambda x: str(x), doc_info[2]))])
            f_inv.write(text)
        length = f_inv.tell() - offset
        catalog[term] = {'t': [offset, length]}
    f_inv.close()
    with open('../data/CATALOG_'+str(cnt)+'.json', 'wb') as f_cat:
        json.dump(catalog, f_cat)
    return

TAG_DOC = '<DOC>'
TAG_DOCNO_START, TAG_DOCNO_END = '<DOCNO>', '</DOCNO>'
TAG_TEXT_START, TAG_TEXT_END = '<TEXT>', '</TEXT>'
BATCH = 1000
cnt, batch_cnt = 0, 0
term_dict, term_map, doc_map = {}, {}, {}
term_id = {}
vocabulary, documents, ttf = 0, 0, 0

stem_method = 'as_is'
now = datetime.now()
for file_name in file_list:
    if file_name[:2].lower() != 'ap':
        continue
    with open(dir_data+file_name) as f:
        for l in f:
            line = l.replace('\n','')
            if line[:len(TAG_DOC)] == TAG_DOC:
                if batch_cnt == BATCH:
                    dumpFile(term_dict, cnt) # dump file for a doc batch
                    batch_cnt = 0
                    term_dict = {}
                elif batch_cnt > 0:
                    vocabulary, documents, ttf = updateTerm(text, docno,
                        term_dict, term_id,
                        term_map, doc_map, stem_method, vocabulary, documents, ttf)
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
    vocabulary, documents, ttf = updateTerm(text, docno, term_dict, term_id,
                    term_map, doc_map, stem_method, vocabulary, documents, ttf)
    dumpFile(term_dict, cnt)
print 'total number of documents loaded is', cnt
print 'running time is ', datetime.now() - now

print 'writing STATS and MAP files'
now = datetime.now()
stats = {'V': vocabulary, 'D': documents, 'TTF': ttf}
with open('../data/STATS.json', 'wb') as f:
    json.dump(stats, f)
with open('../data/TERM_MAP.json', 'wb') as f:
    json.dump(term_map, f)
with open('../data/DOC_MAP.json', 'wb') as f:
    json.dump(doc_map, f)
print 'running time is ', datetime.now() - now
