from stemming.porter2 import stem
import re
import cPickle

def loadStopWords():
    stop_words = set()
    with open('stoplist.txt', 'rb') as f:
        for line in f:
            stop_words.add(line.replace('\n', ''))
    return stop_words

def stemWord(w, stem_method, stop_words = None):
    w = stemAsIs(w)
    if stem_method in ['no_stop_words', 'stemmed_no_stop_words']:
        if w in stop_words:
            return ''
    if stem_method in ['stemmed', 'stemmed_no_stop_words']:
        w = stem(w)
    return w

def stemAsIs(w):
    w = w.strip('=.-:\\').replace(',','').lower()
    w = w.rstrip('\'s')
    return w

def tokenizer(text):
    return re.findall(r"[a-zA-Z](?:[a-zA-Z'/-])+|\w+(?:['.,]?\w+)*",text)

def loadTFInfo(tf_line):
    '''input: 'term_id.df.ttf.|doc_id tf pos,pos\n',
    output: term_id, {'df': df, 'ttf': ttf, 'info': [[doc_id, tf, [pos]]]}
    '''
    doc_info = {}
    slices = tf_line.rstrip('\n').split('.')
    term_id, doc_info['df'], doc_info['ttf'] = map(lambda x: int(x), slices[:3])
    docs = slices[-1].split('|')[1:]
    doc_info['info'] = map(lambda x: loadDocInfo(x), docs)
    return term_id, doc_info

def loadDocInfo(doc_line): # 'doc_id tf pos,pos'
    doc = doc_line.split(' ')
    pos = map(lambda x: int(x), doc[2].split(','))
    doc = map(lambda x: int(x), doc[:2])
    doc.append(pos)
    return doc

def dumpFile(term_dict, cnt, dir_file):
    # term_dict: {term: {'df': df, 'ttf': ttf, 'info':[[docid, tf, [pos]]]}}
    f_inv = open(dir_file + 'INV_'+str(cnt)+'.txt', 'wb')
    catalog = {}
    for term, term_info in term_dict.iteritems():
        offset, length = dumpTerm(term, term_info, f_inv)
        catalog[term] = {'t': [offset, length]}
    f_inv.close()
    with open(dir_file + 'CATALOG_'+str(cnt), 'wb') as f_cat:
        cPickle.dump(catalog, f_cat)
    return

def dumpTerm(term, term_info, f_inv):
    # term_info: {'df': df, 'ttf': ttf, 'info':[[docid, tf, [pos]]]}
    offset = f_inv.tell()
    sorted_info = sorted(term_info['info'], key = lambda x: -x[1]) # ordered by tf desc
    df_info_text = str(term) + '.' + str(term_info['df']) + '.' + \
                        str(term_info['ttf']) + '.'
    f_inv.write(df_info_text)
    for doc_info in sorted_info:
        text = '|' + ' '.join([str(doc_info[0]), str(doc_info[1]),
                                ','.join(map(lambda x: str(x), doc_info[2]))])
        f_inv.write(text)
    f_inv.write('\n')
    length = f_inv.tell() - offset
    return offset, length
