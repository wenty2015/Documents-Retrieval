import os
import sys
from datetime import datetime
from func import *

def loadBatchFile(f, batch_catalog, batch_inv):
    index_no = f.split('_')[1]
    batch_catalog.append(loadDict(DIR, f, [0, 1, 2]))
    batch_inv.append(open(DIR + INV_FILE + '_' + index_no + '.txt', 'rb'))
    return

def dumpTermWithLoc(term, term_info, doc_blocks, f_inv):
    # term_info: {'df': df, 'ttf': ttf, 'loc':[[batch_loc, loc]]}
    offset = f_inv.tell()
    df_info_text = str(term) + '.' + str(term_info['df']) + '.' + \
                        str(term_info['ttf'])
    f_inv.write(df_info_text)
    for loc_info in term_info['loc']:
        idx, loc = loc_info
        doc_info = doc_blocks[idx][loc]
        text = '|' + ' '.join([str(doc_info[0]), str(doc_info[1]),
                                ','.join(map(lambda x: str(x), doc_info[2]))])
        f_inv.write(text)
    f_inv.write('\n')
    length = f_inv.tell() - offset
    return offset, length

def mergeBatchFile(batch_order, batch_catalog, batch_inv):
    merged_catalog = {}
    merged_file = open(DIR_MERGE + INV_FILE + '_' + str(batch_order) + '.txt', 'wb')

    for i, catalog in enumerate(batch_catalog):
        for term in catalog.keys():
            # term_dict: {'df': df, 'ttf': ttf, 'loc':[[batch_loc, loc]]}
            term_dict = {'df': 0, 'ttf': 0, 'loc':[]}
            doc_blocks = []
            for j in xrange(i, len(batch_catalog)):
                cat, inv_file = batch_catalog[j], batch_inv[j]
                if term in cat:
                    offset, length = cat[term]
                    inv_file.seek(offset) # check
                    tf_line = inv_file.read(length)
                    term_id, doc_info = loadTFInfo(tf_line)
                    term_dict['df'] += doc_info['df']
                    term_dict['ttf'] += doc_info['ttf']
                    doc_blocks.append(doc_info['info'])
                    if j > i:
                        del cat[term]
            term_dict['loc'] = mergeSort(doc_blocks) #[[batch_loc, loc]]
            merged_offset, merged_length = dumpTermWithLoc(
                        term, term_dict, doc_blocks, merged_file)
            merged_catalog[term] = [merged_offset, merged_length]
        batch_inv[i].close()
    merged_file.close()
    print 'V', len(merged_catalog.keys())
    dumpDict(DIR_MERGE, CAT_FILE + '_' + str(batch_order), merged_catalog)
    return

def mergeSort(sorted_lists):
    # sorted_lists: [ [ [doc_id, tf, [pos]], ], ]
    total_num = sum(map(lambda l: len(l), sorted_lists))
    merged_list = [ [] for i in xrange(total_num) ]
    pointer_list = [0] * len(sorted_lists)
    for loc in xrange(total_num):
        max_cnt = 0
        for i, p in enumerate(pointer_list):
            if p < len(sorted_lists[i]):
                if sorted_lists[i][p][1] > max_cnt:
                    max_index = i
                    max_cnt = sorted_lists[i][p][1]
        merged_list[loc] = (max_index, pointer_list[max_index])
        pointer_list[max_index] += 1
    return merged_list

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
