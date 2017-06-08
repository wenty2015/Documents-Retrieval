import os
import cPickle
import sys
from datetime import datetime
from func import *

def loadBatchFile(f, batch_catalog, batch_inv):
    index_no = f.split('_')[1]
    with open(DIR + f, 'rb') as f:
        batch_catalog.append(cPickle.load(f))
    batch_inv.append(open(DIR + INV_FILE + '_' + index_no + '.txt', 'rb'))
    return

def mergeBatchFile(batch_order, batch_catalog, batch_inv):
    merged_file = open(DIR_MERGE + INV_FILE + '_' + str(batch_order) + '.txt', 'wb')
    merged_catalog = {}
    for i, catalog in enumerate(batch_catalog):
        for term in catalog.keys():
            merged_offset = merged_file.tell()
            for j in xrange(i, len(batch_catalog)):
                cat, inv_file = batch_catalog[j], batch_inv[j]
                if term in cat:
                    offset, length = cat[term]['t'] # text information
                    present_offset = offset
                    while present_offset < offset + length:
                        inv_file.seek(present_offset) # check
                        info = inv_file.readline()
                        present_offset = inv_file.tell()
                        merged_file.write(info)
                    if j > i:
                        del cat[term]
            merged_length = merged_file.tell() - 1 - merged_offset
            merged_catalog[term] = {'t': [merged_offset, merged_length]}
        batch_inv[i].close()
    merged_file.close()
    print 'V', len(merged_catalog.keys())
    with open(DIR_MERGE + CAT_FILE + '_' + str(batch_order), 'wb') as f:
        cPickle.dump(merged_catalog, f)
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
file_list = sorted(filter(lambda f: f[:len(CAT_FILE)] == CAT_FILE, os.listdir(DIR)),
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
