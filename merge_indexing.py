import os
import sys
from datetime import datetime
from func import *

def loadBatchFile(f, batch_catalog, batch_inv):
    index_no = f.split('_')[1]
    batch_catalog.append(loadDict(DIR, f, [0, 1, 2]))
    batch_inv.append(open(DIR + INV_FILE + '_' + index_no + '.txt', 'rb'))
    return

def mergeBatchFile(batch_order, batch_catalog, batch_inv):
    merged_catalog = {}
    merged_file = open(DIR_MERGE + INV_FILE + '_' + str(batch_order) + '.txt', 'wb')

    for i, catalog in enumerate(batch_catalog):
        for term in catalog.keys():
            # term_dict: {'df': df, 'ttf': ttf, 'info':[[docid, tf, [pos]]]}
            term_dict = {'df': 0, 'ttf': 0, 'info':[]}
            for j in xrange(i, len(batch_catalog)):
                cat, inv_file = batch_catalog[j], batch_inv[j]
                if term in cat:
                    offset, length = cat[term]
                    inv_file.seek(offset) # check
                    tf_line = inv_file.readline()
                    term_id, doc_info = loadTFInfo(tf_line)
                    term_dict['df'] += doc_info['df']
                    term_dict['ttf'] += doc_info['ttf']
                    term_dict['info'].extend(doc_info['info'])
                    if j > i:
                        del cat[term]
            merged_offset, merged_length = dumpTerm(term, term_dict, merged_file)
            merged_catalog[term] = [merged_offset, merged_length]
        batch_inv[i].close()
    merged_file.close()
    print 'V', len(merged_catalog.keys())
    dumpDict(DIR_MERGE, CAT_FILE + '_' + str(batch_order), merged_catalog)
    return

args = sys.argv
if len(args) == 1:
    stem_method = 'no_stop_words'
else:
    stem_method = args[1]

DIR_DATA = '../data/' + stem_method + '/'
DIR = DIR_DATA + 'indexing_files/'
DIR_MERGE = DIR_DATA + 'merged_indexing_files/'
CAT_FILE = 'CATALOG'
INV_FILE = 'INV'
BATCH = 100
file_list = filter(lambda f: f[:len(CAT_FILE)] == CAT_FILE, os.listdir(DIR))
file_list = sorted(map(lambda f: f.rstrip('.txt'), file_list),
                    key = lambda x: int(x.split('_')[1]))
print len(file_list), 'files to load'

now = datetime.now()
batch_cnt, batch_order = 0, 1
batch_catalog, batch_inv = [], []
for f in file_list:
    batch_cnt += 1
    loadBatchFile(f, batch_catalog, batch_inv)
    if batch_cnt == BATCH or f == file_list[-1]:
        mergeBatchFile(batch_order, batch_catalog, batch_inv)
        batch_catalog, batch_inv = [], []
        batch_cnt, batch_order = 0, batch_order + 1
print 'running time is ', datetime.now() - now
