from stemming.porter2 import stem
import re
'''from nltk.stem.porter import *
STEMMER = PorterStemmer()
'''
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
        # w = STEMMER.stem(w)
    return w

def stemAsIs(w):
    # w = w.strip('=.-:\\').replace(',','').lower()
    return w.lower()

def tokenizer(text):
    # r"[a-zA-Z](?:[a-zA-Z'/-])+|\w+(?:['.,]?\w+)*"
    return re.findall(r"\w+(?:\.?\w+)*",text)

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

def dumpDict(file_dir, file_name, d, list_flag = True):
    f = open(file_dir + file_name + '.txt', 'wb')
    for key, value in d.iteritems():
        line_list = [key] + value if list_flag else [key, value]
        line = ' '.join(map(lambda t: str(t), line_list)) + '\n'
        f.write(line)
    f.close()
    return

def loadDict(file_dir, file_name, int_loc_list = []):
    f = open(file_dir + file_name + '.txt', 'rb')
    d =  {}
    for line in f:
        line_list = line.rstrip('\n').split(' ')
        for loc in int_loc_list:
            line_list[loc] = int(line_list[loc])
        d[line_list[0]] = line_list[1:] if len(line_list[1:]) > 1 else line_list[1]
    f.close()
    return d
