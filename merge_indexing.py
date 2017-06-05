import os
import cPickle

def loadBatchFile(f, batch_catalog, batch_inv):
    index_no = f.split('_')[1]
    with open(DIR + f, 'rb') as f:
        batch_catalog.append(cPickle.load(f))
    batch_inv.append(open(DIR + INV_FILE + '_' + index_no + '.txt', 'rb'))
    return

def mergeBatchFile(f, batch_catalog, batch_inv):
    index_no = f.split('_')[1]
    merged_file = open(DIR_MERGE + INV_FILE + '_' + index_no + '.txt', 'wb')
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
                        inv_file.seek(offset)
                        info = inv_file.readline()
                        present_offset = inv_file.tell()
                        merged_file.write(info)
                    if j > i:
                        del cat[term]
            merged_length = merged_file.tell() - 1 - merged_offset
            merged_catalog[term] = [merged_offset, merged_length]
        batch_inv[i].close()
    merged_file.close()
    with open(DIR_MERGE + f, 'wb') as f:
        cPickle.dump(merged_catalog, f)
    return

DIR = '../data/indexing_files/'
DIR_MERGE = '../data/merged_indexing_files/'
CAT_FILE = 'CATALOG'
INV_FILE = 'INV'
BATCH = 10
file_list = filter(lambda f: f[:len(CAT_FILE)] == CAT_FILE, os.listdir(DIR))
print len(file_list), 'files to load'

batch_cnt = 0
batch_catalog, batch_inv = [], []
for f in file_list:
    batch_cnt += 1
    loadBatchFile(f, batch_catalog, batch_inv)
    if batch_cnt == BATCH or f == file_list[-1]:
        mergeBatchFile(f, batch_catalog, batch_inv)
        batch_catalog, batch_inv = [], []
